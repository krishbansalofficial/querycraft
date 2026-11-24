"""Agent loop (v2 part 3): retrieve tables, write SQL, run it, and repair on an error or an empty result.

A small deterministic state machine chooses the tools; the model only writes SQL, which is the one skill it was
fine tuned for (the SFT model never saw tool call syntax, so free form tool choice would be an untrained format).
After a failed attempt the model sees its query plus the observation and writes a new one.

    retrieve_tables -> [model writes SQL] -> run_sql -> ok (non empty, or empty allowed) -> finish
                                               |-> error: describe_table(tables the query used), widen schema
                                               |-> empty: sample_values(columns compared with string literals)
                                               '-> [model rewrites SQL with the observation], at most max_repairs
                                                   times and max_steps tool calls in total

Tools are plain functions over a read only SQLite connection with the timeout and row cap of execution.py.

    python -m querycraft.agent configs/agent/agent_1.5b.yaml     # tune on val, then val/dev/held back
"""
import json
import random
import re
import sys
import time
from pathlib import Path

import sqlglot
from opentelemetry import trace
from sqlglot import exp

from .config import load_config, logged_stage, run_dir
from .data import load_split, selection
from .execution import execute
from .prompts import PromptBuilder, extract_sql, gold_tables
from .schema import _iq, read_schema, serialize

TRACER = trace.get_tracer("querycraft.agent")
ROW_CAP = 1000          # rows returned to a caller; scoring still compares full results (execution.py)
SQL_TIMEOUT = 30.0      # same as the scoring harness
TOOLS = ("retrieve_tables", "describe_table", "sample_values", "run_sql", "finish")


# ---------------------------------------------------------------- tools

def describe_table(db_path, name):
    t = [t for t in read_schema(db_path) if t["name"].lower() == name.lower()]
    return serialize(t) if t else f"no table named {name}"


def sample_values(db_path, table, column, like=None, n=10):
    """Distinct values of one column; values resembling `like` first (case insensitive)."""
    q, c = _iq(table), _iq(column)
    out = []
    if like:
        pat = "'%" + str(like).strip("%").replace("'", "''") + "%'"
        rows, _ = execute(db_path, f"SELECT DISTINCT {c} FROM {q} WHERE {c} LIKE {pat} LIMIT {n}", 10.0)
        out += [r[0] for r in rows or []]
    rows, err = execute(db_path, f"SELECT DISTINCT {c} FROM {q} WHERE {c} IS NOT NULL LIMIT {n}", 10.0)
    out += [r[0] for r in rows or [] if r[0] not in out]
    return out[:n]


def run_sql(db_path, sql, timeout=SQL_TIMEOUT, max_rows=ROW_CAP, columns=None):
    if not sql.strip():
        return None, "empty query"
    return execute(db_path, sql, timeout, max_rows, columns)


def is_empty(rows):
    """No rows, or only NULLs (an aggregate over zero matching rows returns one NULL row)."""
    return not rows or all(v is None for r in rows for v in r)


def literal_columns(sql, tables, limit=3):
    """(table, column, string literal) for columns compared with a string literal (=, LIKE, IN)."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return []
    names = {t["name"].lower(): t for t in tables}
    alias = {}
    for t in tree.find_all(exp.Table):
        alias[t.alias_or_name.lower()] = t.name.lower()
    out = []
    for node in tree.find_all(exp.EQ, exp.Like, exp.In):
        col, lit = node.find(exp.Column), node.find(exp.Literal)
        if not (col and lit and lit.is_string):
            continue
        tn = alias.get(col.table.lower()) if col.table else next(
            (n for n in alias.values() if n in names and any(c["name"].lower() == col.name.lower()
                                                            for c in names[n]["columns"])), None)
        if tn in names:
            cname = next((c["name"] for c in names[tn]["columns"] if c["name"].lower() == col.name.lower()), None)
            if cname and (names[tn]["name"], cname, lit.this) not in out:
                out.append((names[tn]["name"], cname, lit.this))
    return out[:limit]


# ---------------------------------------------------------------- model backends

class OfflineLLM:
    """vLLM in process (experiments). complete(prompts, t) -> [(text, prompt_tokens, completion_tokens)]."""

    def __init__(self, sql_cfg):
        from .infer import Generator
        self.gen = Generator(sql_cfg, sql_cfg["model"])
        self.tok, self.seed, self.max_tokens = self.gen.tok, sql_cfg["seed"], sql_cfg["eval"]["max_new_tokens"]

    def complete(self, prompts, temperature=0.0, seed=None):
        from vllm import SamplingParams
        if not prompts:
            return []
        sp = SamplingParams(temperature=temperature, seed=self.seed if seed is None else seed, max_tokens=self.max_tokens)
        outs = self.gen.llm.generate(prompts, sp)
        return [(o.outputs[0].text, len(o.prompt_token_ids), len(o.outputs[0].token_ids)) for o in outs]


class OpenAILLM:
    """vLLM's OpenAI compatible server (serving). Same prompt text as offline, sent to /v1/completions."""

    def __init__(self, base_url, model, tokenizer, max_tokens=256, seed=0, workers=32):
        from transformers import AutoTokenizer
        self.base, self.model, self.max_tokens, self.seed, self.workers = base_url.rstrip("/"), model, max_tokens, seed, workers
        self.tok = AutoTokenizer.from_pretrained(tokenizer)

    def _one(self, prompt, temperature, seed):
        import httpx
        r = httpx.post(f"{self.base}/v1/completions", timeout=120, json={
            "model": self.model, "prompt": prompt, "max_tokens": self.max_tokens, "temperature": temperature,
            "seed": seed})
        r.raise_for_status()
        d = r.json()
        return d["choices"][0]["text"], d["usage"]["prompt_tokens"], d["usage"]["completion_tokens"]

    def complete(self, prompts, temperature=0.0, seed=None):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(lambda p: self._one(p, temperature, self.seed if seed is None else seed), prompts))

    def sample(self, prompt, n, temperature):
        """n samples of one prompt in one request -> ([texts], total tokens)."""
        import httpx
        r = httpx.post(f"{self.base}/v1/completions", timeout=120, json={
            "model": self.model, "prompt": prompt, "max_tokens": self.max_tokens, "temperature": temperature,
            "n": n, "seed": self.seed})
        r.raise_for_status()
        d = r.json()
        return [c["text"] for c in d["choices"]], d["usage"]["total_tokens"]


# ---------------------------------------------------------------- the loop

def _log(st, tool, t0, **attrs):
    ms = round(1000 * (time.perf_counter() - t0), 2)
    st["trace"].append({"tool": tool, "ms": ms, **attrs})
    with TRACER.start_as_current_span(f"agent.{tool}") as span:  # one span per step (no-op without a provider)
        span.set_attribute("question_id", str(st["ex"]["id"]))
        span.set_attribute("ms", ms)
        for k, v in attrs.items():
            span.set_attribute(k, v if isinstance(v, (int, float, bool, str)) else json.dumps(v)[:500])


def _observe(st, sql, rows, err, opts):
    """Build the observation for a failed attempt and record the inspection tools used. Returns (text, widen)."""
    db, tables = st["ex"]["db_path"], read_schema(st["ex"]["db_path"])
    if err is not None:
        widen = opts["widen_on_error"] and ("no such table" in err or "no such column" in err)
        parts = [f"The query failed with: {err}"]
        if widen:
            for t in sorted(gold_tables(sql))[:2]:
                if st["steps"] >= opts["max_steps"] - 2:
                    break
                t0 = time.perf_counter()
                d = describe_table(db, t)
                st["steps"] += 1
                _log(st, "describe_table", t0, table=t)
                if not d.startswith("no table"):
                    parts.append(f"Table {t}:\n{d[:1500]}")
        return "\n".join(parts), widen
    parts = ["The query ran but returned no rows (or only NULL)."]
    for table, col, lit in literal_columns(sql, tables):
        if st["steps"] >= opts["max_steps"] - 2:
            break
        t0 = time.perf_counter()
        vals = sample_values(db, table, col, like=lit)
        st["steps"] += 1
        _log(st, "sample_values", t0, table=table, column=col, n=len(vals))
        parts.append(f"Column {table}.{col} has values like: {', '.join(repr(v)[:40] for v in vals)}")
    return "\n".join(parts), False


def run_batch(llm, rows, tables_for, opts, budget=1280):
    """Run the loop for many questions at once, one model call per round. Returns per question records."""
    pb_link = PromptBuilder(llm.tok, budget, linking="retriever")
    pb_full = PromptBuilder(llm.tok, budget)
    states = []
    for ex in rows:
        st = {"ex": {**ex, "link_tables": []}, "trace": [], "steps": 0, "tokens": 0, "attempts": [], "done": False,
              "widen": False, "extra": None, "t0": time.perf_counter()}
        t0 = time.perf_counter()
        st["ex"]["link_tables"] = sorted(tables_for(ex))
        st["steps"] += 1
        _log(st, "retrieve_tables", t0, tables=st["ex"]["link_tables"])
        states.append(st)
    for rnd in range(opts["max_repairs"] + 1):
        live = [s for s in states if not s["done"]]
        if not live:
            break
        prompts = [(pb_full if s["widen"] else pb_link).prompt_text(s["ex"], extra=s["extra"]) for s in live]
        t0 = time.perf_counter()
        outs = llm.complete(prompts)
        gen_ms = 1000 * (time.perf_counter() - t0) / len(live)
        redo = []
        for s, p, (text, pt, ct) in zip(live, prompts, outs):
            s["tokens"] += pt + ct
            sql = extract_sql(text)
            if sql in [a["sql"] for a in s["attempts"]]:
                redo.append((s, p))
            s["pending"] = sql
            s["gen_ms"] = s.get("gen_ms", 0) + gen_ms
        if redo:  # a repeat of an earlier attempt: sample once instead (seeded), so a retry can differ
            outs = llm.complete([p for _, p in redo], temperature=opts["retry_temperature"], seed=rnd + 1)
            for (s, _), (text, pt, ct) in zip(redo, outs):
                s["tokens"] += pt + ct
                s["pending"] = extract_sql(text)
        for s in live:
            sql = s.pop("pending")
            t0 = time.perf_counter()
            res, err = run_sql(s["ex"]["db_path"], sql)
            s["steps"] += 1
            _log(s, "run_sql", t0, ok=err is None, rows=len(res or []), error=(err or "")[:200])
            empty = err is None and is_empty(res)
            s["attempts"].append({"sql": sql, "error": err, "n_rows": None if err else (0 if empty else len(res))})
            good = err is None and (not empty or not opts["repair_on_empty"])
            if good or rnd == opts["max_repairs"] or s["steps"] >= opts["max_steps"] - 1:
                s["done"] = True
                continue
            obs, widen = _observe(s, sql, res, err, opts)
            s["widen"] = s["widen"] or widen
            s["extra"] = f"Your previous query:\n{sql}\n{obs}\nWrite a corrected SQLite query. Reply with only the SQL query."
    out = []
    for s in states:
        ok = [a for a in s["attempts"] if a["error"] is None]
        final = (next((a for a in reversed(ok) if a["n_rows"]), None) or (ok[-1] if ok else None)
                 or s["attempts"][-1])["sql"]
        t0 = time.perf_counter()
        s["steps"] += 1
        _log(s, "finish", t0, sql=final[:500])
        ex = s["ex"]
        out.append({"id": ex["id"], "db_id": ex["db_id"], "db_path": ex["db_path"], "difficulty": ex.get("difficulty"),
                    "question": ex["question"], "gold_sql": ex.get("sql", ""), "pred_sql": final,
                    "tables": ex["link_tables"], "steps": s["steps"], "tokens": s["tokens"],
                    "repairs": len(s["attempts"]) - 1, "attempts": s["attempts"], "trace": s["trace"],
                    "ms_batched": round(1000 * (time.perf_counter() - s["t0"]), 1)})
    return out


# ---------------------------------------------------------------- experiments

def _ranked_tables(cfg, split):
    """Tables per question from the trained retriever's val chosen setting (precomputed ranked lists)."""
    from .retriever import pick
    chosen = json.loads(Path(cfg["retriever_sweep"]).read_text())["chosen"]
    ranked = {(d := json.loads(l))["id"]: d for l in Path(cfg["ranked"].format(split=split)).read_text().splitlines()}
    return lambda ex: pick(ranked[ex["id"]][chosen["source"]], read_schema(ex["db_path"]), chosen["k"], chosen["hop"])


def options(cfg, **over):
    return {"max_steps": cfg["max_steps"], "max_repairs": cfg["max_repairs"],
            "retry_temperature": cfg["retry_temperature"], **cfg["defaults"], **over}


def _run_split(cfg, llm, split, opts, out_path):
    from .evaluate import score_file
    rows = load_split(split)
    if cfg.get("limit"):
        rows = rows[: cfg["limit"]]
    t0 = time.time()
    recs = run_batch(llm, rows, _ranked_tables(cfg, split), opts)
    wall = time.time() - t0
    Path(out_path).write_text("".join(json.dumps(r) + "\n" for r in recs))
    m = score_file(out_path)
    n = len(recs)
    m |= {"mean_steps": round(sum(r["steps"] for r in recs) / n, 2), "mean_tokens": round(sum(r["tokens"] for r in recs) / n, 1),
          "mean_repairs": round(sum(r["repairs"] for r in recs) / n, 3), "share_repaired": round(sum(r["repairs"] > 0 for r in recs) / n, 4),
          "amortized_ms_per_question": round(1000 * wall / n, 1)}
    return m


@selection
def tune(cfg, llm):
    """Pick repair_on_empty and widen_on_error by val EX (ties: fewer mean steps)."""
    d = run_dir(cfg) / "tune"
    d.mkdir(exist_ok=True)
    res = {}
    for roe in cfg["grid"]["repair_on_empty"]:
        for woe in cfg["grid"]["widen_on_error"]:
            t = f"empty{int(roe)}_widen{int(woe)}"
            res[t] = {"repair_on_empty": roe, "widen_on_error": woe,
                      **{k: v for k, v in _run_split(cfg, llm, "val", options(cfg, repair_on_empty=roe, widen_on_error=woe),
                                                     d / f"val_{t}.jsonl").items() if k in ("ex", "mean_steps", "mean_tokens")}}
            print(t, res[t], flush=True)
    best = max(res.values(), key=lambda r: (r["ex"], -r["mean_steps"]))
    return {"by_setting": res, "chosen": {k: best[k] for k in ("repair_on_empty", "widen_on_error")}}


def latency_sample(cfg, llm, opts, split="dev"):
    """Unbatched, one question at a time: wall clock latency of the agent and of a single pass (first attempt)."""
    rows = load_split(split)
    rows = random.Random(cfg["seed"]).sample(rows, cfg["latency_sample"])
    tables_for = _ranked_tables(cfg, split)
    agent, single = [], []
    for ex in rows:
        t0 = time.perf_counter()
        run_batch(llm, [ex], tables_for, opts)
        agent.append(1000 * (time.perf_counter() - t0))
        t0 = time.perf_counter()
        run_batch(llm, [ex], tables_for, {**opts, "max_repairs": 0})
        single.append(1000 * (time.perf_counter() - t0))

    def q(xs, p):
        xs = sorted(xs)
        return round(xs[min(len(xs) - 1, int(p * len(xs)))], 1)
    return {"n": len(rows), "agent_ms": {"mean": round(sum(agent) / len(agent), 1), "p50": q(agent, 0.5), "p90": q(agent, 0.9)},
            "single_pass_ms": {"mean": round(sum(single) / len(single), 1), "p50": q(single, 0.5), "p90": q(single, 0.9)}}


def main(cfg_path):
    cfg = load_config(cfg_path)
    sql_cfg = load_config(cfg["sql_config"])
    out = run_dir(cfg)
    with logged_stage(cfg, "eval") as meta:
        llm = OfflineLLM(sql_cfg)
        tuned = tune(cfg, llm) if cfg.get("grid") else {"chosen": {}}
        opts = options(cfg, **tuned["chosen"])
        res = {"name": cfg["name"], "model": sql_cfg["model"], "options": opts, "tune": tuned}
        for split in cfg.get("splits", ("val", "dev", "heldback")):
            res[split] = _run_split(cfg, llm, split, opts, out / f"{split}_agent.jsonl")
            print(split, {k: res[split][k] for k in ("ex", "mean_steps", "mean_tokens", "share_repaired")}, flush=True)
        if cfg.get("latency_sample"):
            res["latency"] = latency_sample(cfg, llm, opts)
        (out / "results.json").write_text(json.dumps(res, indent=2))
        meta["results"] = "results.json"


if __name__ == "__main__":
    main(sys.argv[1])
