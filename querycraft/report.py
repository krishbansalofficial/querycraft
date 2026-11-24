"""Build README result tables + plots strictly from logged runs (runs/*/results.json and *_meta.json).
Planned runs without results are listed under "Targets (not yet measured)"; nothing is ever filled in by hand."""
import json
import math
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
    paths = Path("configs").glob("[lrg]*/*.yaml")  # local + rented, then grpo, linking, retriever (built on SFT) last
    for cfg_path in sorted(paths, key=lambda p: (p.parent.name in ("grpo", "linking", "retriever"), str(p))):
        cfg = load_config(cfg_path)
        if cfg.get("kind"):  # v2 component configs (retriever, agent, serve) are not SQL model rows
            continue
        d = RUNS_DIR / cfg["name"]
        res, tm, em = _j(d / "results.json"), _j(d / "train_meta.json"), _j(d / "eval_meta.json")
        rows.append({"cfg": cfg, "profile": cfg_path.parent.name, "res": res, "train": tm, "eval": em, "dir": d})
    return rows


def _size(model):
    return re.search(r"(\d+(?:\.\d+)?B)", model, re.I).group(1).upper()


def _label(r):
    c = r["cfg"]
    t = c.get("train")
    sft = f"{_size(c['model'])} SFT {'DoRA' if 'dora' in c['model'] else 'LoRA'}"
    if c["prompt"].get("linking") == "oracle":
        return f"{sft}, oracle tables (upper bound)"
    if c["prompt"].get("linking") == "selector":
        return f"{sft}, trained table selector (2 pass)"
    if c["prompt"].get("linking") == "retriever":
        return f"{sft}, trained table retriever (2 pass)"
    if c["prompt"].get("linking") == "self":
        return f"{sft}, self linked tables (2 pass)"
    if not t:
        return f"{_size(c['model'])} {'few shot (3)' if c['prompt'].get('shots') else 'zero shot'}"
    if t["method"] == "grpo":
        return f"{_size(c['model'])} SFT LoRA + GRPO ({t['reward']} reward)"
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


def _ex_by_id(r, split="dev"):
    p = r["dir"] / "eval" / f"{split}_{r['res'].get('selected_checkpoint', 'base')}.scored.jsonl"
    return {(d := json.loads(l))["id"]: d["ex"] for l in p.read_text().splitlines()}


def _paired(a, b, split="dev"):
    """(only a right, only b right) over the questions both runs scored."""
    xa, xb = _ex_by_id(a, split), _ex_by_id(b, split)
    ids = xa.keys() & xb.keys()
    return sum(xa[i] and not xb[i] for i in ids), sum(xb[i] and not xa[i] for i in ids)


def _sign_p(a, b):
    n, k = a + b, min(a, b)
    if n and 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n < 0.001:
        return "<0.001"
    return round(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n), 3) if n else 1.0


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
            for keys in dict.fromkeys(tuple(r["res"]["val_ex_by_checkpoint"]) for r in sel):  # SFT and GRPO save at different steps
                out.append("| Model | " + " | ".join(keys) + " | Trainable params | Peak VRAM (GB) |")
                out.append("|---|" + "---|" * (len(keys) + 2))
                for r in (r for r in sel if tuple(r["res"]["val_ex_by_checkpoint"]) == keys):
                    v = r["res"]["val_ex_by_checkpoint"]
                    cells = [f"**{x}**" if k == r["res"]["selected_checkpoint"] else str(x) for k, x in v.items()]
                    tp = r["train"].get("trainable_params") if r["train"] else None
                    out.append(f"| {_label(r)} | " + " | ".join(cells) + f" | {tp or 'all'} | {r['train'].get('peak_vram_gb', '') if r['train'] else ''} |")
                out.append("")
        zs = {_size(r["cfg"]["model"]): r for r in done if not r["cfg"].get("train") and not r["cfg"]["prompt"].get("shots") and not r["cfg"]["prompt"].get("linking")}
        ft = [r for r in done if r["cfg"].get("train") and r["cfg"]["train"]["method"] != "grpo" and _size(r["cfg"]["model"]) in zs]
        if ft:
            out.append("### Does fine tuning beat zero shot? Paired against the same size zero shot model\n")
            out.append("p is an exact two sided sign test on the questions where exactly one of the two models is right "
                       "(dev: 1,534 questions, held back: 100).\n")
            out.append("| Model | Zero shot dev EX | Dev EX | Only FT right | Only zero shot right | p | "
                       "Zero shot held back | Held back | Only FT right | Only zero shot right | p |")
            out.append("|---|---|---|---|---|---|---|---|---|---|---|")
            for r in ft:
                z = zs[_size(r["cfg"]["model"])]
                cells = []
                for split in ("dev", "heldback"):
                    a, b = _paired(z, r, split)
                    cells += [z["res"][split]["ex"], r["res"][split]["ex"], b, a, _sign_p(a, b)]
                out.append(f"| {_label(r)} | " + " | ".join(map(str, cells)) + " |")
            out.append("")
        built = [r for r in done if r["cfg"]["model"].startswith("runs/")]  # GRPO and schema linking, on the SFT model
        if built:
            out.append("### Built on an SFT model, paired against it\n")
            out.append("GRPO arms start from the 1.5B LoRA r16 SFT checkpoint and train 125 steps on 1,000 prompts the "
                       "SFT model solves 1 to 7 times out of 8. Schema linking runs use an SFT model unchanged (1.5B LoRA r16, or the 7B DoRA run) and "
                       "only change which tables the prompt shows (`docs/plans/schema_linking.md`); the oracle row uses "
                       "the gold SQL's tables and is an upper bound, not a usable system. p is an exact two sided sign "
                       "test on the questions where exactly one of the two models is right.\n")
            out.append("| Model | SFT dev EX | Dev EX | Only new right | Only SFT right | p | "
                       "SFT held back | Held back | Only new right | Only SFT right | p |")
            out.append("|---|---|---|---|---|---|---|---|---|---|---|")
            for r in built:
                parent = next(x for x in done if x["cfg"]["name"] == Path(r["cfg"]["model"]).parent.parent.name)
                cells = []
                for split in ("dev", "heldback"):
                    a, b = _paired(parent, r, split)
                    cells += [parent["res"][split]["ex"], r["res"][split]["ex"], b, a, _sign_p(a, b)]
                out.append(f"| {_label(r)} | " + " | ".join(map(str, cells)) + " |")
            out.append("")
        conf =[(r, json.loads(f.read_text())) for r in done if (f := r["dir"] / "confidence" / "confidence.json").exists()]
        if conf:
            out.append("### Knowing when it is wrong: voting and abstention (BIRD dev)\n")
            out.append("8 sampled queries per question are executed and grouped by result; the largest group is the "
                       "answer and its share is the confidence. Temperature and thresholds are chosen on val; the "
                       "threshold columns answer only questions at or above the val threshold that kept 50% or 70% "
                       "of val questions.\n")
            out.append("| Model | Single answer EX | Vote of 8 EX | Confidence AUROC | Answers at val 50% threshold | "
                       "EX there | Answers at val 70% threshold | EX there | Held back EX at val 50% threshold |")
            out.append("|---|---|---|---|---|---|---|---|---|")
            for r, c in conf:
                d, h = c["dev"]["at_val_thresholds"], c["heldback"]["at_val_thresholds"]
                out.append(f"| {_label(r)} | {r['res']['dev']['ex']} | {c['dev']['vote_ex']} | {c['dev']['auroc']} | "
                           f"{d['0.5']['coverage']}% | {d['0.5']['accuracy']} | {d['0.7']['coverage']}% | "
                           f"{d['0.7']['accuracy']} | {h['0.5']['accuracy']} ({h['0.5']['coverage']}% answered) |")
            out.append("")
    out += retrieval_section(done)
    out += abstain_section() + agent_section() + bench_section() + iowa_section()
    if todo:
        out.append("### Targets (not yet measured)\n")
        out.append("These runs are configured but have no logged results yet. No numbers are reported for them.\n")
        for r in todo:
            out.append(f"* {_label(r)} (`{r['cfg']['_path']}`, {'A100 80GB' if r['profile'] == 'rented' or r['cfg'].get('budget_hours') else 'RTX 4070 Laptop 8GB'})")
        out.append("")
    return "\n".join(out), done


def gap_recovered(sft, oracle, new, n_boot=2000, seed=0):
    """Share of the oracle's dev gain a linker recovers, (new - sft) / (oracle - sft), with a paired bootstrap
    95% interval over questions."""
    import random
    a, o, b = _ex_by_id(sft), _ex_by_id(oracle), _ex_by_id(new)
    ids = sorted(a.keys() & o.keys() & b.keys())

    def share(s):
        den = sum(o[i] - a[i] for i in s)
        return (sum(b[i] - a[i] for i in s) / den) if den else float("nan")

    rng = random.Random(seed)
    boots = sorted(share([rng.choice(ids) for _ in ids]) for _ in range(n_boot))
    return round(100 * share(ids), 1), round(100 * boots[int(0.025 * n_boot)], 1), round(100 * boots[int(0.975 * n_boot) - 1], 1)


def retrieval_section(done):
    """How well each way of picking tables keeps the gold tables (val and dev), and the oracle gap recovered."""
    from .data import load_split
    from .linker import set_metrics
    from .prompts import fk_hop, gold_tables
    from .schema import read_schema
    ret = _j(RUNS_DIR / "table_retriever" / "results.json")
    if not ret:
        return []
    sel = _j(RUNS_DIR / "qwen1.5b_table_selector" / "results.json")
    ch = ret["chosen"]
    chosen_tag = f"{ch['source']}_k{ch['k']}{'_hop' if ch['hop'] else ''}"
    names = {"bi": "bi encoder", "rerank": "bi encoder + cross encoder rerank"}
    out = ["### Picking the tables: set metrics (v2 part 1)\n",
           "Gold tables come from the gold SQL. \"Every gold table kept\" is the share of questions whose prompt can "
           "contain everything the answer needs. Self linking and the selector are the v1 rows; the retriever's "
           f"source, k and foreign key hop were chosen by second pass val EX (chosen: **{chosen_tag}**).\n",
           "| Method | Split | Recall | Precision | Every gold table kept | Mean tables shown |",
           "|---|---|---|---|---|---|"]
    for split in ("val", "dev"):
        exs = load_split(split)
        f = RUNS_DIR / "qwen1.5b_lora_r16" / "eval" / f"{split}_checkpoint-420.jsonl"
        rows = []
        if f.exists():
            pred = {(d := json.loads(l))["id"]: d["pred_sql"] for l in f.read_text().splitlines()}
            rows.append(("Self linking (first pass tables + 1 FK hop)", set_metrics(
                [(fk_hop(read_schema(ex["db_path"]), gold_tables(pred[ex["id"]])), gold_tables(ex["sql"]))
                 for ex in exs if ex["id"] in pred])))
        if sel:
            rows.append(("Trained table selector (lists tables)", sel[split]))
        for t, m in ret[f"{split}_by_setting"].items():
            if t == chosen_tag or not t.endswith("_hop"):
                src, k = t.split("_k")[0], t.split("_k")[1].split("_")[0]
                label = f"Retriever, {names[src]}, top {k}" + (" + 1 FK hop" if t.endswith("_hop") else "")
                rows.append((f"**{label}**" if t == chosen_tag else label, m))
        for label, m in rows:
            out.append(f"| {label} | {split} | {m['recall']} | {m['precision']} | {m['all_gold_kept']} | {m['mean_predicted']} |")
    by = {r["cfg"]["name"]: r for r in done}
    trio = [by.get(n) for n in ("qwen1.5b_lora_r16", "qwen1.5b_lora_r16_oracle", "qwen1.5b_lora_r16_retrieved")]
    if all(trio):
        s, lo, hi = gap_recovered(*trio)
        sel_row = by.get("qwen1.5b_lora_r16_selector")
        extra = ""
        if sel_row:
            s2, lo2, hi2 = gap_recovered(trio[0], trio[1], sel_row)
            extra = f" The trained selector recovers {s2}% [{lo2}, {hi2}]."
        out.append(f"\nOn dev the retriever second pass recovers **{s}%** of the oracle's gain over the SFT model "
                   f"(paired bootstrap 95% interval [{lo}, {hi}], 2,000 resamples of the 1,534 questions).{extra}\n")
    out.append("")
    return out


def _p(p):
    return f"p < {str(p)[1:]}" if str(p).startswith("<") else f"p = {p}"


def _scored(path):
    p = Path(path)
    return {(d := json.loads(l))["id"]: d["ex"] for l in p.read_text().splitlines()} if p.exists() else None


def _vs(new, base):
    """(new EX, base EX, only new right, only base right, p) over shared questions; None if either is missing."""
    if not new or not base:
        return None
    ids = new.keys() & base.keys()
    a = sum(new[i] and not base[i] for i in ids)
    b = sum(base[i] and not new[i] for i in ids)
    return (round(100 * sum(new[i] for i in ids) / len(ids), 2), round(100 * sum(base[i] for i in ids) / len(ids), 2),
            a, b, _sign_p(a, b), len(ids))


SFT_15 = "runs/qwen1.5b_lora_r16/eval/{split}_checkpoint-420.scored.jsonl"
SYSTEMS = {  # agent run -> (label, plain single pass, retrieved single pass)
    "agent_qwen1.5b": ("1.5B SFT LoRA", SFT_15, "runs/qwen1.5b_lora_r16_retrieved/eval/{split}_base.scored.jsonl"),
    "agent_qwen7b": ("7B SFT DoRA (T4, fp16)", "runs/qwen7b_dora_t4_single/eval/{split}_base.scored.jsonl",
                     "runs/qwen7b_dora_t4_retrieved/eval/{split}_base.scored.jsonl"),
}


def _sft_path(single, split):
    if split == "iowa" and single == SFT_15:
        return "runs/qwen1.5b_sft_iowa/eval/iowa_base.scored.jsonl"
    return single.format(split=split)


def abstain_section():
    r = _j(RUNS_DIR / "abstain_lgbm" / "results.json")
    if not r:
        return []
    f = r["fit"]
    out = ["### Answer or abstain: learned model against the vote share threshold (v2 part 2)\n",
           f"Both rank the same answers (the 1.5B LoRA r16 vote of 8). LightGBM on {len(r['features'])} features "
           f"({', '.join(r['features'])}), fit on val ({f['n_val']} questions); 5 fold CV Brier chose "
           f"{f['params']} with {'Platt' if f['calibration'] == 'sigmoid' else 'isotonic'} calibration. Brier and ECE "
           "(10 bins) score the confidence as a probability that the answer is right; vote share is used as is.\n",
           "| Split | Score | Brier | ECE | AUROC | Answered at val 70% threshold | EX there |", "|---|---|---|---|---|---|---|"]
    for split in ("dev", "heldback"):
        for side, label in (("learned", "learned (calibrated)"), ("baseline", "vote share")):
            s = r[split][side]
            t = s["at_val_thresholds"]["0.7"]
            out.append(f"| {split} | {label} | {s['brier']} | {s['ece']} | {s['auroc']} | {t['coverage']}% | {t['accuracy']} |")
    a = r["dev"]["at_70"]
    out.append(f"\nAt exactly 70% of dev answered ({a['answered']} questions, ties broken by a seeded shuffle): learned "
               f"**{a['learned_acc']}** against vote share **{a['baseline_acc']}** (only learned right {a['only_learned_right']}, "
               f"only vote share right {a['only_baseline_right']}, exact sign test p = {round(a['p'], 3)}; accuracy "
               f"difference 95% bootstrap interval [{a['diff_ci95'][0]}, {a['diff_ci95'][1]}] points).\n")
    ag = _scored(RUNS_DIR / "agent_qwen1.5b" / "dev_agent.scored.jsonl")
    sf = RUNS_DIR / "abstain_lgbm" / "dev_scores.jsonl"
    if ag and sf.exists():  # the API returns the agent's SQL with this confidence: is it calibrated for those answers?
        from .abstain import brier, ece
        from .confidence import auroc
        p = {(d := json.loads(l))["id"]: d["p"] for l in sf.read_text().splitlines()}
        ids = [i for i in p if i in ag]
        ps, ys = [p[i] for i in ids], [int(ag[i]) for i in ids]
        out.append(f"Scored against the 1.5B agent's answers instead (what the API returns), on {len(ids)} dev questions: "
                   f"Brier {brier(ps, ys)}, ECE {ece(ps, ys)}, AUROC {auroc(ps, ys)}.\n")
    return out


def agent_section():
    out = []
    for run, (label, single, retrieved) in SYSTEMS.items():
        r = _j(RUNS_DIR / run / "results.json")
        if not r:
            continue
        if not out:
            out = ["### Agent loop against single pass (v2 part 3)\n",
                   "The agent retrieves tables, writes SQL, runs it, and on an error or an empty result inspects the "
                   "schema or samples column values and rewrites (at most 2 rewrites, 8 tool calls). Options chosen on "
                   "val. p: exact sign test on the questions exactly one system gets right.\n",
                   "| Model | Split | Agent EX | Single pass EX | p | Retrieved single pass EX | p | Mean steps | "
                   "Mean tokens | Share repaired |", "|---|---|---|---|---|---|---|---|---|---|"]
        for split in ("val", "dev", "heldback", "iowa"):
            if split not in r:
                continue
            ag = _scored(RUNS_DIR / run / f"{split}_agent.scored.jsonl")
            v1, v2 = _vs(ag, _scored(_sft_path(single, split))), _vs(ag, _scored(retrieved.format(split=split)))
            s = r[split]
            out.append(f"| {label} | {split} | {s['ex']} | {v1[1] if v1 else 'TBD'} | {v1[4] if v1 else ''} | "
                       f"{v2[1] if v2 else 'TBD'} | {v2[4] if v2 else ''} | {s['mean_steps']} | {s['mean_tokens']} | "
                       f"{s['share_repaired']} |")
        lat = r.get("latency")
        if lat:
            out.append(f"\n{label}, {lat['n']} dev questions one at a time: agent p50 {lat['agent_ms']['p50']} ms "
                       f"(p90 {lat['agent_ms']['p90']}), single pass p50 {lat['single_pass_ms']['p50']} ms "
                       f"(p90 {lat['single_pass_ms']['p90']}).\n")
    return out + [""] if out else []


def bench_section():
    b = _j(RUNS_DIR / "bench_qwen1.5b" / "results.json")
    acc = _vs(_scored("runs/qwen1.5b_lora_r16_awq/eval/dev_base.scored.jsonl"), _scored(SFT_15.format(split="dev")))
    if not b and not acc:
        return []
    out = ["### Serving: bf16 against 4 bit AWQ, 1.5B on the RTX 4070 Laptop (v2 part 4)\n"]
    if acc:
        out.append(f"Dev EX: AWQ {acc[0]} against bf16 {acc[1]} (only AWQ right {acc[2]}, only bf16 right {acc[3]}, "
                   f"exact sign test {_p(acc[4])}, {acc[5]} questions).\n")
    if b:
        out += [f"Locust, {b['seconds_per_level']} s per level, dev prompts, greedy, 256 max new tokens. Cost per 1,000 "
                "queries assumes this throughput at public on demand prices (" +
                ", ".join(f"{g} ${p}/h" for g, p in b["gpu_price_per_hour"].items()) + ").\n",
                "| Variant | Concurrency | Requests/s | Completion tokens/s | p50 ms | p99 ms | Peak GPU MiB | Model GiB | "
                "KV cache tokens | $ per 1k queries (L4) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for v in b["variants"]:
            for l in v["levels"]:
                c = v["cost_per_1000_queries_usd"].get("L4", {}).get(str(l["concurrency"]), "")
                out.append(f"| {v['variant']} | {l['concurrency']} | {l['requests_per_s']} | {l['completion_tokens_per_s']} | "
                           f"{l['p50_ms']} | {l['p99_ms']} | {v['peak_gpu_mib']} | {v['model_memory_gib']} | "
                           f"{v['kv_cache_tokens']} | {c} |")
    return out + [""]


def iowa_section():
    from .evaluate import wilson
    rows = [("1.5B SFT LoRA, single pass", _scored("runs/qwen1.5b_sft_iowa/eval/iowa_base.scored.jsonl")),
            ("1.5B SFT LoRA, retrieved tables", _scored("runs/qwen1.5b_lora_r16_retrieved/eval/iowa_base.scored.jsonl")),
            ("1.5B agent", _scored("runs/agent_qwen1.5b/iowa_agent.scored.jsonl"))]
    rows = [(l, s) for l, s in rows if s]
    if not rows:
        return []
    out = ["### Live demo warehouse: Iowa Liquor Sales (v2 part 6)\n",
           "30 questions written for this project with gold SQL validated by execution (`data/warehouse/iowa_questions.json`, "
           "**needs human review**), over 2,000,000 invoice lines (2022-01-02 to 2022-10-19). A second held back set: "
           "scored only. 95% Wilson interval; p is an exact sign test against the single pass row.\n",
           "| System | EX [95% CI] | Only this right | Only single pass right | p |", "|---|---|---|---|---|"]
    base = rows[0][1]
    for label, s in rows:
        k = sum(s.values())
        lo, hi = wilson(k, len(s))
        v = _vs(s, base)
        out.append(f"| {label} | {round(100 * k / len(s), 2)} [{lo}, {hi}] | {v[2]} | {v[3]} | {v[4]} |")
    return out + [""]


def headline():
    """The README's opening table: one number per part, each from runs/ (TBD when not measured)."""
    def g(path, *keys):
        d = _j(Path(path))
        for k in keys:
            d = (d or {}).get(k) if isinstance(d, dict) else None
        return d
    ret = g(RUNS_DIR / "table_retriever" / "results.json", "dev")
    rows = [("Every gold table kept, dev (trained retriever vs v1 selector)",
             f"{ret['all_gold_kept']} vs {g(RUNS_DIR / 'qwen1.5b_table_selector' / 'results.json', 'dev', 'all_gold_kept')}"
             f" ({ret['mean_predicted']} vs {g(RUNS_DIR / 'qwen1.5b_table_selector' / 'results.json', 'dev', 'mean_predicted')} tables shown)"
             if ret else "TBD")]
    v = _vs(_scored("runs/qwen1.5b_lora_r16_retrieved/eval/dev_base.scored.jsonl"), _scored(SFT_15.format(split="dev")))
    rows.append(("Dev EX, 1.5B SFT: retrieved tables vs full pruned schema", f"{v[0]} vs {v[1]} ({_p(v[4])})" if v else "TBD"))
    for run, (label, single, retrieved) in SYSTEMS.items():
        ag = _scored(RUNS_DIR / run / "dev_agent.scored.jsonl")
        v, w = _vs(ag, _scored(single.format(split="dev"))), _vs(ag, _scored(retrieved.format(split="dev")))
        rows.append((f"Dev EX, {label}: agent vs single pass (vs its own retrieved single pass)",
                     f"{v[0]} vs {v[1]} ({_p(v[4])}); vs {w[1]} ({_p(w[4])})" if v and w else "TBD"))
    a = g(RUNS_DIR / "abstain_lgbm" / "results.json", "dev", "at_70")
    rows.append(("Dev accuracy at 70% answered: learned abstain vs vote share",
                 f"{a['learned_acc']} vs {a['baseline_acc']} (p = {round(a['p'], 3)})" if a else "TBD"))
    v = _vs(_scored("runs/qwen1.5b_lora_r16_awq/eval/dev_base.scored.jsonl"), _scored(SFT_15.format(split="dev")))
    rows.append(("Dev EX, 1.5B 4 bit AWQ vs bf16", f"{v[0]} vs {v[1]} ({_p(v[4])})" if v else "TBD"))
    b = _j(RUNS_DIR / "bench_qwen1.5b" / "results.json")
    if b:
        for v in b["variants"]:
            l1 = next(l for l in v["levels"] if l["concurrency"] == 1)
            lmax = max(v["levels"], key=lambda l: l["concurrency"])
            rows.append((f"1.5B {v['variant']} on the laptop: p50 latency at 1 user; $ per 1k queries at {lmax['concurrency']} users (L4 price)",
                         f"{l1['p50_ms']} ms; ${v['cost_per_1000_queries_usd']['L4'][str(lmax['concurrency'])]}"))
    else:
        rows.append(("Latency and cost per 1k queries", "TBD"))
    return "| Measure | Result |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows) + "\n"


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
    conf = [(r, json.loads(f.read_text())) for r in done if (f := r["dir"] / "confidence" / "confidence.json").exists()]
    if conf:
        fig, ax = plt.subplots(figsize=(6, 4))
        for r, c in conf:
            pts = [(p["coverage"], p["accuracy"]) for p in c["dev"]["curve"]]
            ax.plot(*zip(*pts), marker="o", ms=3, label=_label(r) + " (vote share)")
        ab = _j(RUNS_DIR / "abstain_lgbm" / "results.json")
        if ab:  # v2: learned, calibrated abstain model over the same 1.5B LoRA r16 vote answers
            pts = sorted((p["coverage"], p["accuracy"]) for p in ab["dev"]["learned"]["curve"] if p["accuracy"] is not None)
            ax.plot(*zip(*pts), "k--", lw=1.6, label="1.5B LoRA (bf16) r16, learned abstain model (v2)")
            ax.axvline(70, color="grey", lw=0.6, ls=":")
        ax.set_xlabel("Questions answered (%)")
        ax.set_ylabel("Accuracy on answered questions (%)")
        ax.set_title("Accuracy vs coverage on BIRD dev (vote of 8)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / "accuracy_vs_coverage.png", dpi=130)
        made.append("accuracy_vs_coverage.png")
    grpo = [r for r in done if (r["cfg"].get("train") or {}).get("method") == "grpo" and (r["dir"] / "train_log.json").exists()]
    if grpo:  # reward hacking check: does the trained reward rise without execution correctness rising with it?
        fig, ax = plt.subplots(figsize=(6, 4))
        for r in grpo:
            arm = r["cfg"]["train"]["reward"]
            log = [x for x in json.loads((r["dir"] / "train_log.json").read_text()) if "reward" in x]
            for key, style in ((f"rewards/{arm}_reward/mean", "-"), ("rewards/binary_reward/mean", ":")):
                ys = [x[key] for x in log if key in x]
                if arm == "binary" and style == ":":
                    continue  # same series as the trained reward
                ys = [sum(ys[max(0, i - 9):i + 1]) / len(ys[max(0, i - 9):i + 1]) for i in range(len(ys))]
                ax.plot(range(1, len(ys) + 1), ys, style, label=f"{arm} arm: {key.split('/')[1].replace('_', ' ')}")
        ax.set_xlabel("GRPO step")
        ax.set_ylabel("Mean train reward (10 step moving average)")
        ax.set_title("Trained reward vs execution correctness")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / "grpo_rewards.png", dpi=130)
        made.append("grpo_rewards.png")
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
        s = re.sub(r"(<!-- HEADLINE:START -->).*(<!-- HEADLINE:END -->)", lambda m: f"{m.group(1)}\n{headline()}{m.group(2)}", s, flags=re.S)
        README.write_text(s)
    print(md)


if __name__ == "__main__":
    main()
