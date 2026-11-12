"""Ask questions of any SQLite database, entirely on this machine. The model answers only when confident.

    pip install -r requirements-demo.txt
    python -m querycraft.demo configs/local/11_qwen1.5b_lora_r16.yaml     # then open http://127.0.0.1:7860

Uses the run's selected checkpoint and the temperature and threshold chosen on val by querycraft.confidence.
Databases are opened read only with a timeout, and the server listens on localhost only.
"""
import json
import sys

import gradio as gr

from .config import load_config, run_dir
from .confidence import vote
from .data import load_split
from .execution import execute
from .infer import Generator
from .prompts import PromptBuilder, extract_sql
from .run_eval import _serving

SQLITE_MAGIC = b"SQLite format 3\x00"
MAX_ROWS = 200


def build(cfg_path, target="0.7"):  # threshold that answered 70% of val questions
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
    dbs = {r["db_id"]: r["db_path"] for r in load_split("dev")}

    def ask(db_id, upload, question, hint):
        path = upload or dbs.get(db_id)
        if not path or not question.strip():
            return "Pick a database (or upload one) and type a question.", "", None
        with open(path, "rb") as f:
            if f.read(16) != SQLITE_MAGIC:
                return "That file is not a SQLite database.", "", None
        prompt = pb.prompt_text({"db_path": path, "question": question, "evidence": hint or ""})
        greedy = extract_sql(gen.generate([prompt], adapter)[0])
        samples = [extract_sql(s) for s in gen.generate([prompt], adapter, n=n, temperature=temp)[0]]
        sql, c, k = vote({"db_path": path, "samples": samples, "greedy": greedy})
        cols = []
        rows, err = execute(path, sql, timeout=15, max_rows=MAX_ROWS, columns=cols)
        votes = f"{round(c * n)}/{n} sampled queries agree"
        if err:
            status = f"### ⚠️ Could not answer\nThe best query failed: `{err}` ({votes})."
        elif c >= threshold:
            status = f"### ✅ Answer\n{votes} (needs {round(threshold * n)}/{n} to answer; {k} distinct results)."
        else:
            status = (f"### ⚠️ Not sure\nOnly {votes} (needs {round(threshold * n)}/{n}). The model's candidate "
                      f"answers disagree, so treat the result below as a guess.")
        table = [list(r) for r in rows] if rows else None
        return status, sql, (gr.update(value=table, headers=cols) if table else None)

    with gr.Blocks(title="Querycraft") as ui:
        gr.Markdown(f"# Querycraft\nAsk a question about a SQLite database. Model: `{cfg['name']}` "
                    f"({tag}). Everything runs locally; databases are opened read only.")
        with gr.Row():
            db = gr.Dropdown(sorted(dbs), label="BIRD dev database", value="california_schools")
            up = gr.File(label="...or upload your own .sqlite", file_types=[".sqlite", ".db", ".sqlite3"], type="filepath")
        q = gr.Textbox(label="Question", placeholder="Which school has the highest average math SAT score?")
        h = gr.Textbox(label="Hint (optional)", placeholder="e.g. 'eligible free rate = Free Meal Count / Enrollment'")
        go = gr.Button("Ask", variant="primary")
        status = gr.Markdown()
        sql = gr.Code(label="SQL", language="sql")
        out = gr.Dataframe(label=f"Result (first {MAX_ROWS} rows)", wrap=True)
        go.click(ask, [db, up, q, h], [status, sql, out])
        q.submit(ask, [db, up, q, h], [status, sql, out])
    return ui


if __name__ == "__main__":
    build(sys.argv[1]).launch(server_name="127.0.0.1", server_port=7860, max_file_size="500mb")
