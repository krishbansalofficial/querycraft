"""Forgetting probe: Python coding ability (HumanEval + MBPP sanitized, pass@1, greedy).

The base is a *Coder* model, so general Python coding is the capability SQL specialisation is most
likely to erode, and it is scored by execution like our main metric. Generated code runs in a
separate interpreter process with a timeout inside a temp dir.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset

_CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def _code(text):
    m = _CODE.findall(text)
    return max(m, key=len) if m else text


def problems():
    out = []
    for r in load_dataset("openai/openai_humaneval", split="test"):
        out.append({"suite": "humaneval", "id": r["task_id"],
                    "user": "Complete the following Python function. Return the complete function in a ```python code block.\n"
                            f"```python\n{r['prompt']}```",
                    "prefix": r["prompt"], "entry": r["entry_point"],
                    "test": f"{r['test']}\ncheck({r['entry_point']})\n"})
    for r in load_dataset("google-research-datasets/mbpp", "sanitized", split="test"):
        tests = "\n".join(r["test_list"])
        out.append({"suite": "mbpp", "id": f"mbpp/{r['task_id']}",
                    "user": f"{r['prompt']}\nYour code should pass these tests:\n{tests}\n"
                            "Return the Python code in a ```python code block.",
                    "prefix": "\n".join(r["test_imports"]), "entry": None, "test": tests + "\n"})
    return out


def program(p, completion):
    code = _code(completion)
    if p["suite"] == "humaneval" and f"def {p['entry']}" not in code:
        code = p["prefix"] + code  # model continued the body only
    elif p["suite"] == "mbpp":
        code = p["prefix"] + "\n" + code
    return code + "\n\n" + p["test"]


def run_program(src, timeout=10):
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "prog.py")
        with open(f, "w") as fh:
            fh.write(src)
        try:
            r = subprocess.run([sys.executable, "-I", f], cwd=d, capture_output=True, timeout=timeout)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def run(gen, adapter=None, out_path=None):
    probs = problems()
    if gen.cfg["eval"].get("limit"):  # smoke tests: first N of each suite
        lim = gen.cfg["eval"]["limit"]
        probs = [p for s in ("humaneval", "mbpp") for p in [q for q in probs if q["suite"] == s][:lim]]
    prompts = [gen.tok.apply_chat_template([{"role": "user", "content": p["user"]}], tokenize=False,
                                           add_generation_prompt=True) for p in probs]
    outs = gen.generate(prompts, adapter, max_tokens=768)
    with ThreadPoolExecutor(16) as pool:
        ok = list(pool.map(run_program, [program(p, o) for p, o in zip(probs, outs)]))
    res = {}
    for suite in ("humaneval", "mbpp"):
        xs = [k for p, k in zip(probs, ok) if p["suite"] == suite]
        res[suite] = {"n": len(xs), "pass@1": round(100 * sum(xs) / len(xs), 2)}
    res["combined"] = {"n": len(ok), "pass@1": round(100 * sum(ok) / len(ok), 2)}
    if out_path:
        with open(out_path, "w") as f:
            for p, o, k in zip(probs, outs, ok):
                f.write(json.dumps({"id": p["id"], "passed": k, "completion": o}) + "\n")
    return res
