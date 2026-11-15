"""Ask questions of any SQLite database, entirely on this machine. The model answers only when confident.

    python -u -m querycraft.demo configs/local/11_qwen1.5b_lora_r16.yaml  # then open http://127.0.0.1:7860

Uses the run's selected checkpoint and the temperature and threshold chosen on val by querycraft.confidence.
Databases are opened read only with a timeout, and the server listens on localhost only. The page itself is
querycraft/demo_ui.html (no build step, no frontend dependencies); this file is a small JSON API around the model.
"""
import json
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SQLITE_MAGIC = b"SQLite format 3\x00"
MAX_ROWS = 200
MAX_UPLOAD = 500 * 2**20
UI = Path(__file__).with_name("demo_ui.html")


def load_examples():
    """{db_id: (db_path, [up to 3 example dev questions])} for the database picker."""
    from .data import load_split
    out = {}
    for r in load_split("dev"):
        path, qs = out.setdefault(r["db_id"], (r["db_path"], []))
        if len(qs) < 3:
            qs.append({"question": r["question"], "hint": r.get("evidence", "")})
    return out


def results_table():
    """Every finished run's headline numbers, straight from runs/*/results.json (same source as the README)."""
    from .report import _label, collect
    rows = []
    for r in collect():
        if r["res"] and r["eval"] and r["eval"].get("status") == "ok":
            dev = r["res"].get("dev", {})
            rows.append({"name": r["cfg"]["name"], "label": _label(r), "dev": dev.get("ex"), "ci": dev.get("ex_ci95"),
                         "heldback": r["res"].get("heldback", {}).get("ex")})
    return rows


def build_answerer(cfg_path, target="0.7"):  # threshold that answered 70% of val questions
    from .confidence import vote
    from .config import load_config, run_dir
    from .execution import execute
    from .infer import Generator
    from .prompts import PromptBuilder, extract_sql
    from .run_eval import _serving

    cfg = load_config(cfg_path)
    rdir = run_dir(cfg)
    tag = json.loads((rdir / "results.json").read_text()).get("selected_checkpoint", "base")
    cf = rdir / "confidence" / "confidence.json"
    conf = json.loads(cf.read_text()) if cf.exists() else {}
    temp, n = conf.get("temperature", 0.8), conf.get("n", 8)
    threshold = (conf.get("thresholds_from_val") or {}).get(target) or 0.5
    model_path, adapter = _serving(cfg, rdir / "ckpt" / tag if tag != "base" else None)
    gen = Generator(cfg, model_path)
    pb = PromptBuilder(gen.tok, cfg["prompt"]["schema_budget"])
    lock = threading.Lock()  # one GPU, one question at a time

    def ask(path, question, hint):
        prompt = pb.prompt_text({"db_path": path, "question": question, "evidence": hint})
        with lock:
            greedy = extract_sql(gen.generate([prompt], adapter)[0])
            samples = [extract_sql(s) for s in gen.generate([prompt], adapter, n=n, temperature=temp)[0]]
        sql, c, k = vote({"db_path": path, "samples": samples, "greedy": greedy})
        cols = []
        rows, err = execute(path, sql, timeout=15, max_rows=MAX_ROWS, columns=cols)
        state = "error" if err else "empty" if not rows else "answer" if c >= threshold else "unsure"
        return {"state": state, "sql": sql, "error": err, "columns": cols, "rows": [list(r) for r in rows or []],
                "votes": {"agree": round(c * n), "n": n, "needed": round(threshold * n), "distinct": k}}

    info = {"model": cfg["name"], "checkpoint": tag, "n": n, "needed": round(threshold * n)}
    return ask, info


def make_handler(ask, info, examples, uploads_dir):
    uploads = {}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):  # keep the terminal quiet
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, UI.read_bytes(), "text/html; charset=utf-8")
            if self.path == "/api/info":
                return self._send(200, {**info, "databases": {k: v[1] for k, v in sorted(examples.items())}})
            if self.path == "/api/results":
                return self._send(200, results_table())
            self._send(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            if self.path == "/api/upload":
                if n > MAX_UPLOAD:
                    return self._send(413, {"error": "File is larger than 500 MB."})
                data = self.rfile.read(n)
                if not data.startswith(SQLITE_MAGIC):
                    return self._send(400, {"error": "That file is not a SQLite database."})
                token = uuid.uuid4().hex
                uploads[token] = Path(uploads_dir) / f"{token}.sqlite"
                uploads[token].write_bytes(data)
                return self._send(200, {"db": f"upload:{token}"})
            if self.path != "/api/ask" or n > 1 << 20:
                return self._send(404, {"error": "not found"})
            req = json.loads(self.rfile.read(n) or b"{}")
            db, question = str(req.get("db", "")), str(req.get("question", "")).strip()
            path = uploads.get(db[7:]) if db.startswith("upload:") else (examples.get(db) or (None,))[0]
            if not path or not question:
                return self._send(400, {"error": "Pick a database and type a question."})
            t0 = time.time()
            try:
                out = ask(str(path), question, str(req.get("hint", "")))
            except Exception as e:  # surface the failure to the page instead of a dropped connection
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})
            self._send(200, {**out, "seconds": round(time.time() - t0, 1)})

    return Handler


def serve(ask, info, examples, port=7860):
    with tempfile.TemporaryDirectory() as tmp:
        server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(ask, info, examples, tmp))
        print(f"Querycraft demo on http://127.0.0.1:{port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    ask, info = build_answerer(sys.argv[1])
    serve(ask, info, load_examples())
