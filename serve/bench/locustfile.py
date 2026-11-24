"""Load test for vLLM's OpenAI compatible server: each user sends dev prompts back to back (no think time).

    QC_PROMPTS=runs/bench/prompts.jsonl QC_MODEL=querycraft QC_TOKENS_OUT=tokens.json \
      locust -f serve/bench/locustfile.py --headless -u 8 -r 8 -t 60s --host http://127.0.0.1:8001 --csv out

Greedy, 256 max new tokens, the exact prompt text the evaluation uses. Completion tokens are summed so the
benchmark can report tokens per second next to requests per second.
"""
import itertools
import json
import os
import threading

from locust import HttpUser, events, task

PROMPTS = [json.loads(l)["prompt"] for l in open(os.environ.get("QC_PROMPTS", "runs/bench/prompts.jsonl"))]
MODEL = os.environ.get("QC_MODEL", "querycraft")
_cycle, _lock = itertools.cycle(range(len(PROMPTS))), threading.Lock()
TOKENS = {"completion": 0, "prompt": 0, "ok": 0}


class Client(HttpUser):
    wait_time = staticmethod(lambda: 0)

    @task
    def complete(self):
        with _lock:
            i = next(_cycle)
        with self.client.post("/v1/completions", json={"model": MODEL, "prompt": PROMPTS[i], "max_tokens": 256,
                                                       "temperature": 0.0}, catch_response=True, name="completion") as r:
            if r.status_code == 200:
                u = r.json()["usage"]
                with _lock:
                    TOKENS["completion"] += u["completion_tokens"]
                    TOKENS["prompt"] += u["prompt_tokens"]
                    TOKENS["ok"] += 1
            else:
                r.failure(r.text[:200])


@events.test_stop.add_listener
def _dump(**_):
    with open(os.environ.get("QC_TOKENS_OUT", "tokens.json"), "w") as f:
        json.dump(TOKENS, f)
