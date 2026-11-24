"""Querycraft API (v2 part 4). FastAPI in front of vLLM's OpenAI compatible server.

    vllm serve Qwen/Qwen2.5-Coder-1.5B-Instruct --enable-lora --lora-modules querycraft=runs/qwen1.5b_lora_r16/ckpt/checkpoint-420
    uvicorn serve.app:app --port 8000          # or: docker compose -f serve/docker-compose.yml up

POST /ask      {question, database, hint} -> sql, columns, rows, confidence, abstained, tables, steps, tokens
GET  /health   service + model server status
GET  /metrics  Prometheus
GET  /databases
GET  /         the v1 demo page (querycraft/demo_ui.html), served through /api/* adapters onto /ask

Per question: retrieve tables (trained retriever, CPU), agent loop (querycraft/agent.py), 8 samples for the vote
features, learned abstain model -> calibrated probability; abstain below the val threshold that answered 70% of
val. Every question is logged to SQLite for the drift report (querycraft/drift.py). Databases open read only.
"""
import json
import os
import sqlite3
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from opentelemetry import trace
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)  # run and data paths are relative to the repo
VLLM_URL = os.environ.get("QC_VLLM_URL", "http://127.0.0.1:8001")
MODEL = os.environ.get("QC_MODEL", "querycraft")
TOKENIZER = os.environ.get("QC_TOKENIZER", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
LOG_DB = Path(os.environ.get("QC_LOG_DB", "runs/live/questions.sqlite"))
COVERAGE = os.environ.get("QC_COVERAGE", "0.7")   # answer the share of val questions this threshold kept
N_VOTES, VOTE_T = 8, 0.6                          # as in v1 (temperature chosen on val)
SQLITE_MAGIC = b"SQLite format 3\x00"

ASKS = Counter("qc_asks_total", "Questions answered", ["outcome"])
LAT = Histogram("qc_ask_seconds", "End to end /ask latency", buckets=(0.25, 0.5, 1, 2, 4, 8, 16, 32, 64))
STEPS = Histogram("qc_agent_steps", "Agent tool calls per question", buckets=(2, 3, 4, 5, 6, 7, 8))
TOKENS = Counter("qc_tokens_total", "Prompt + completion tokens")
CONF = Histogram("qc_confidence", "Calibrated confidence", buckets=[i / 10 for i in range(11)])
UP = Gauge("qc_model_up", "1 if the model server answered the last health check")
TRACER = trace.get_tracer("querycraft.api")


def setup_tracing():
    """OTLP traces (one span per agent step) when OTEL_EXPORTER_OTLP_ENDPOINT is set; otherwise none."""
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    tp = TracerProvider(resource=Resource.create({"service.name": "querycraft-api"}))
    tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(tp)


def databases():
    """BIRD dev databases + the warehouse demo (if built) + QC_EXTRA_DBS (name=path,...)."""
    from querycraft.data import load_split
    dbs = {r["db_id"]: r["db_path"] for r in load_split("dev")}
    for p in Path("data/warehouse").glob("*.sqlite"):
        dbs[p.stem] = str(p)
    for item in filter(None, os.environ.get("QC_EXTRA_DBS", "").split(",")):
        k, v = item.split("=", 1)
        dbs[k] = v
    return {k: v for k, v in dbs.items() if Path(v).exists()}


class State:
    pass


S = State()


def init_log():
    LOG_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(LOG_DB) as c:
        c.execute("CREATE TABLE IF NOT EXISTS questions (id TEXT PRIMARY KEY, ts REAL, db TEXT, question TEXT, hint TEXT, "
                  "tables TEXT, sql TEXT, confidence REAL, abstained INT, outcome TEXT, error TEXT, latency_s REAL, "
                  "steps INT, tokens INT, model TEXT)")


def log_question(row):
    with sqlite3.connect(LOG_DB) as c:
        c.execute("INSERT INTO questions VALUES (:id, :ts, :db, :question, :hint, :tables, :sql, :confidence, :abstained, "
                  ":outcome, :error, :latency_s, :steps, :tokens, :model)", row)


def load():
    import joblib

    from querycraft.agent import OpenAILLM, options
    from querycraft.config import load_config
    from querycraft.retriever import Retriever
    rcfg = load_config("configs/retriever/table_retriever.yaml")
    S.retriever = Retriever("runs/table_retriever", rcfg, device="cpu")
    S.chosen = json.loads(Path("runs/table_retriever/sweep.json").read_text())["chosen"]
    acfg = load_config("configs/agent/agent_1.5b.yaml")
    ares = Path("runs/agent_qwen1.5b/results.json")
    S.opts = json.loads(ares.read_text())["options"] if ares.exists() else options(acfg)
    S.llm = OpenAILLM(VLLM_URL, MODEL, TOKENIZER, seed=0)
    ab = Path("runs/abstain_lgbm")
    # joblib unpickles: only ever our own file, written by querycraft.abstain under runs/ (never user supplied)
    S.abstain = joblib.load(ab / "model.joblib") if (ab / "model.joblib").exists() else None
    S.threshold = json.loads((ab / "results.json").read_text())["fit"]["thresholds"][COVERAGE] if S.abstain else None
    S.dbs = databases()
    S.uploads = {}
    S.tmp = tempfile.mkdtemp(prefix="qc_uploads_")
    S.lock = threading.Lock()
    init_log()


@asynccontextmanager
async def lifespan(app):
    setup_tracing()
    load()
    yield


app = FastAPI(title="Querycraft", lifespan=lifespan)


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    database: str
    hint: str = ""


def answer(question, db_name, db_path, hint=""):
    from querycraft.abstain import FEATURES, feature_row
    from querycraft.agent import is_empty, run_batch, run_sql
    from querycraft.confidence import vote
    from querycraft.prompts import PromptBuilder, extract_sql
    from querycraft.retriever import pick
    from querycraft.schema import read_schema
    t0 = time.time()
    ex = {"id": uuid.uuid4().hex, "db_id": db_name, "db_path": db_path, "question": question, "evidence": hint}
    with TRACER.start_as_current_span("ask") as span:
        span.set_attribute("database", db_name)
        ranked = S.retriever.rank_many([ex])[0][S.chosen["source"]]
        tables = pick(ranked, read_schema(db_path), S.chosen["k"], S.chosen["hop"])
        rec = run_batch(S.llm, [ex], lambda e: tables, S.opts)[0]
        conf, votes = None, None
        tokens = rec["tokens"]
        if S.abstain is not None:
            prompt = PromptBuilder(S.llm.tok, 1280).prompt_text(ex)  # the v1 vote prompt the abstain model learned on
            texts, vt = S.llm.sample(prompt, N_VOTES, VOTE_T)
            tokens += vt
            first = rec["attempts"][0]
            chosen, share, k = vote({"db_path": db_path, "samples": [extract_sql(t) for t in texts], "greedy": first["sql"]})
            f = feature_row(ex, {"confidence": share, "n_results": k, "chosen_sql": chosen},
                            {"pred_sql": first["sql"], "pred_error": first["error"]}, [s for _, s in ranked], S.chosen["k"])
            conf = float(S.abstain.predict_proba([[f[x] for x in FEATURES]])[0, 1])
            votes = {"agree": round(share * N_VOTES), "n": N_VOTES, "distinct": k}
        cols = []
        rows, err = run_sql(db_path, rec["pred_sql"], columns=cols)
        abstained = conf is not None and conf < S.threshold
        outcome = "error" if err else "empty" if is_empty(rows) else "abstained" if abstained else "answered"
        span.set_attribute("outcome", outcome)
    out = {"id": ex["id"], "sql": rec["pred_sql"], "columns": cols, "rows": [list(r) for r in rows or []],
           "confidence": None if conf is None else round(conf, 4), "abstained": abstained,
           "threshold": S.threshold, "votes": votes, "tables": sorted(tables), "steps": rec["steps"], "repairs": rec["repairs"],
           "tokens": tokens, "error": err, "outcome": outcome, "latency_s": round(time.time() - t0, 3)}
    ASKS.labels(outcome).inc()
    LAT.observe(out["latency_s"])
    STEPS.observe(rec["steps"])
    TOKENS.inc(tokens)
    if conf is not None:
        CONF.observe(conf)
    log_question({"id": ex["id"], "ts": time.time(), "db": db_name, "question": question, "hint": hint,
                  "tables": json.dumps(out["tables"]), "sql": out["sql"], "confidence": out["confidence"],
                  "abstained": int(abstained), "outcome": outcome, "error": err, "latency_s": out["latency_s"],
                  "steps": rec["steps"], "tokens": tokens, "model": MODEL})
    return out


def _resolve(name):
    path = S.uploads.get(name[7:]) if name.startswith("upload:") else S.dbs.get(name)
    if not path:
        raise HTTPException(404, f"unknown database {name!r}; see GET /databases")
    return str(path)


@app.post("/ask")
def ask(req: AskIn):
    return answer(req.question.strip(), req.database, _resolve(req.database), req.hint)


@app.get("/databases")
def list_dbs():
    from querycraft.schema import read_schema
    return [{"name": k, "tables": [t["name"] for t in read_schema(v)]} for k, v in sorted(S.dbs.items())]


@app.get("/health")
def health():
    import httpx
    try:
        ok = httpx.get(f"{VLLM_URL}/health", timeout=5).status_code == 200
    except Exception:
        ok = False
    UP.set(int(ok))
    return {"status": "ok" if ok else "degraded", "model_server": ok, "model": MODEL,
            "retriever": S.chosen, "abstain_threshold": S.threshold, "databases": len(S.dbs)}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------- v1 demo page, adapted onto /ask

@app.get("/", response_class=HTMLResponse)
def page():
    return (ROOT / "querycraft" / "demo_ui.html").read_text()


@app.get("/api/info")
def info():
    from querycraft.data import load_split
    ex = {}
    for r in load_split("dev"):
        qs = ex.setdefault(r["db_id"], [])
        if len(qs) < 3:
            qs.append({"question": r["question"], "hint": r.get("evidence", "")})
    for k in S.dbs:
        ex.setdefault(k, [])
    return {"model": MODEL, "checkpoint": "agent + retriever + learned abstain", "n": N_VOTES,
            "needed": None, "databases": dict(sorted(ex.items()))}


@app.get("/api/results")
def results():
    from querycraft.demo import results_table
    return results_table()


@app.post("/api/upload")
async def upload(request: Request):
    data = await request.body()
    if len(data) > 500 * 2**20:
        raise HTTPException(413, "File is larger than 500 MB.")
    if not data.startswith(SQLITE_MAGIC):
        return Response(json.dumps({"error": "That file is not a SQLite database."}), 400, media_type="application/json")
    token = uuid.uuid4().hex
    p = Path(S.tmp) / f"{token}.sqlite"
    p.write_bytes(data)
    S.uploads[token] = p
    return {"db": f"upload:{token}"}


@app.post("/api/ask")
def demo_ask(body: dict):
    db, question = str(body.get("db", "")), str(body.get("question", "")).strip()
    if not db or not question:
        return Response(json.dumps({"error": "Pick a database and type a question."}), 400, media_type="application/json")
    try:
        o = answer(question, db, _resolve(db), str(body.get("hint", "")))
    except HTTPException as e:
        return Response(json.dumps({"error": e.detail}), e.status_code, media_type="application/json")
    state = {"answered": "answer", "abstained": "unsure"}.get(o["outcome"], o["outcome"])
    v = o["votes"] or {"agree": 0, "n": N_VOTES, "distinct": 0}
    # needed=None tells the page the decision is the calibrated confidence, not a vote count
    return {"state": state, "sql": o["sql"], "error": o["error"], "columns": o["columns"], "rows": o["rows"][:200],
            "votes": {**v, "needed": None}, "confidence": o["confidence"], "threshold": o["threshold"],
            "seconds": o["latency_s"]}
