"""Build README result tables + plots strictly from logged runs (runs/*/results.json and *_meta.json).
Planned runs without results are listed under "Targets (not yet measured)"; nothing is ever filled in by hand."""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import RUNS_DIR, load_config  # noqa: E402

FIG = Path("docs/figures")
README = Path("README.md")
HELD_DBS = {"superhero", "student_club", "formula_1"}


def _j(p):
    return json.loads(p.read_text()) if p.exists() else None


def collect():
    rows = []
    for cfg_path in sorted(Path("configs").glob("[lr]*/*.yaml")):  # local + rented
        cfg = load_config(cfg_path)
        d = RUNS_DIR / cfg["name"]
        res, tm, em = _j(d / "results.json"), _j(d / "train_meta.json"), _j(d / "eval_meta.json")
        rows.append({"cfg": cfg, "profile": cfg_path.parent.name, "res": res, "train": tm, "eval": em, "dir": d})
    return rows


def _size(model):
    return re.search(r"(\d+(?:\.\d+)?B)", model).group(1)


def _label(r):
    c = r["cfg"]
    t = c.get("train")
    if not t:
        return f"{_size(c['model'])} {'few shot (3)' if c['prompt'].get('shots') else 'zero shot'}"
    rank = r["train"]["config"]["train"]["rank"] if r["train"] else t["rank"]
    name = {"qlora": "QLoRA", "lora": "LoRA (bf16)", "dora": "DoRA (4 bit base)", "full": "full fine tune"}[t["method"]]
    if t["method"] == "full":
        return f"{_size(c['model'])} {name}"
    return f"{_size(c['model'])} {name} " + ("at best sweep rank" if rank == "best" else f"r{rank}")


def _hours(m):
    return f"{m['wall_clock_s'] / 3600:.2f} h" if m and m.get("status") == "ok" else ""


def _dev_on_held_dbs(r):
    """Dev EX restricted to the three databases the held back set uses (contamination comparison)."""
    tag = r["res"].get("selected_checkpoint", "base")
    p = r["dir"] / "eval" / f"dev_{tag}.scored.jsonl"
    if not p.exists():
        return None
    xs = [json.loads(l)["ex"] for l in p.read_text().splitlines()]
    xs = [x for x, l in zip(xs, p.read_text().splitlines()) if json.loads(l)["db_id"] in HELD_DBS]
    return round(100 * sum(xs) / len(xs), 2) if xs else None


def tables(rows):
    done = [r for r in rows if r["res"] and r["eval"] and r["eval"].get("status") == "ok"]
    todo = [r for r in rows if r not in done]
    out = []
    if done:
        out.append("### Main results (BIRD dev, execution accuracy %)\n")
        out.append("| Model | Val EX (selection) | Dev EX [95% CI] | Dev EX (BIRD official) | Simple | Moderate | Challenging | Held back EX | Dev EX on held back DBs | HumanEval | MBPP | Trained on | Train time |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in done:
            res, dev = r["res"], r["res"].get("dev", {})
            bd = dev.get("by_difficulty", {})
            val = max(res["val_ex_by_checkpoint"].values()) if "val_ex_by_checkpoint" in res else res.get("val", {}).get("ex", "")
            fg = res.get("forgetting", {})
            out.append(f"| {_label(r)} | {val} | {dev.get('ex', '')} [{dev.get('ex_ci95', ['', ''])[0]}, {dev.get('ex_ci95', ['', ''])[1]}] | "
                       f"{dev.get('ex_bird', '')} | {bd.get('simple', {}).get('ex', '')} | {bd.get('moderate', {}).get('ex', '')} | "
                       f"{bd.get('challenging', {}).get('ex', '')} | {res.get('heldback', {}).get('ex', '')} | {_dev_on_held_dbs(r) or ''} | "
                       f"{fg.get('humaneval', {}).get('pass@1', '')} | {fg.get('mbpp', {}).get('pass@1', '')} | "
                       f"{(r['train'] or {}).get('gpu', '') or ''} | {_hours(r['train'])} |")
        out.append("\nHeld back EX vs dev EX on the same three databases is the contamination check: a model that memorised "
                   "BIRD dev would score clearly higher on dev questions than on new questions over the same schemas.\n")
        out.append("### Error categories on BIRD dev (count of wrong predictions)\n")
        tags = list(done[0]["res"]["dev"]["errors"])
        out.append("| Model | " + " | ".join(tags) + " |")
        out.append("|---|" + "---|" * len(tags))
        for r in done:
            e = r["res"]["dev"]["errors"]
            out.append(f"| {_label(r)} | " + " | ".join(str(e.get(t, 0)) for t in tags) + " |")
        out.append("")
        sel = [r for r in done if "val_ex_by_checkpoint" in r["res"]]
        if sel:
            out.append("### Checkpoint selection on validation (EX %, chosen in bold)\n")
            out.append("| Model | " + " | ".join(sel[0]["res"]["val_ex_by_checkpoint"]) + " | Trainable params | Peak VRAM (GB) |")
            out.append("|---|" + "---|" * (len(sel[0]["res"]["val_ex_by_checkpoint"]) + 2))
            for r in sel:
                v = r["res"]["val_ex_by_checkpoint"]
                cells = [f"**{x}**" if k == r["res"]["selected_checkpoint"] else str(x) for k, x in v.items()]
                tp = r["train"].get("trainable_params") if r["train"] else None
                out.append(f"| {_label(r)} | " + " | ".join(cells) + f" | {tp or 'all'} | {r['train'].get('peak_vram_gb', '') if r['train'] else ''} |")
            out.append("")
    if todo:
        out.append("### Targets (not yet measured)\n")
        out.append("These runs are configured but have no logged results yet. No numbers are reported for them.\n")
        for r in todo:
            out.append(f"* {_label(r)} (`{r['cfg']['_path']}`, {'A100 80GB' if r['profile'] == 'rented' else 'RTX 4070 Laptop 8GB'})")
        out.append("")
    return "\n".join(out), done


def plots(done):
    FIG.mkdir(parents=True, exist_ok=True)
    made = []
    if done:
        fig, ax = plt.subplots(figsize=(9, 4))
        diffs = ["simple", "moderate", "challenging"]
        w = 0.8 / len(done)
        for i, r in enumerate(done):
            bd = r["res"]["dev"]["by_difficulty"]
            ax.bar([j + i * w for j in range(3)], [bd.get(d, {}).get("ex", 0) for d in diffs], w, label=_label(r))
        ax.set_xticks([j + 0.4 - w / 2 for j in range(3)], diffs)
        ax.set_ylabel("Dev EX (%)")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(FIG / "dev_ex_by_difficulty.png", dpi=130)
        made.append("dev_ex_by_difficulty.png")
    fg = [r for r in done if r["res"].get("forgetting")]
    if len(fg) > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        for r in fg:
            ax.scatter(r["res"]["dev"]["ex"], r["res"]["forgetting"]["combined"]["pass@1"])
            ax.annotate(_label(r), (r["res"]["dev"]["ex"], r["res"]["forgetting"]["combined"]["pass@1"]), fontsize=7)
        ax.set_xlabel("BIRD dev EX (%)")
        ax.set_ylabel("HumanEval + MBPP pass@1 (%)")
        ax.set_title("Learning vs forgetting")
        fig.tight_layout()
        fig.savefig(FIG / "learn_vs_forget.png", dpi=130)
        made.append("learn_vs_forget.png")
    return made


def main():
    rows = collect()
    md, done = tables(rows)
    figs = plots(done)
    md += "\n" + "\n".join(f"![{f}](docs/figures/{f})" for f in figs)
    Path("docs").mkdir(exist_ok=True)
    Path("docs/results.md").write_text(md)
    if README.exists() and RUNS_DIR.name == "runs":  # smoke runs (runs_smoke) never touch the README
        s = README.read_text()
        s = re.sub(r"(<!-- RESULTS:START -->).*(<!-- RESULTS:END -->)", lambda m: f"{m.group(1)}\n{md}\n{m.group(2)}", s, flags=re.S)
        README.write_text(s)
    print(md)


if __name__ == "__main__":
    main()
