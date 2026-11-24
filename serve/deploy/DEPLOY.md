# Deploying Querycraft (guide only; never run by this project)

Everything here costs money once started. The project itself only ever ran on a laptop and free credits.

## What runs

| Service | Port | Role |
|---|---|---|
| vllm | 8001 | Qwen2.5 Coder 1.5B Instruct + the LoRA r16 adapter (`querycraft`), OpenAI compatible |
| api | 8000 | `serve/app.py`: retriever (CPU), agent loop, learned abstain model, SQLite question log, `/metrics` |
| prometheus | 9090 | scrapes api and vllm |
| grafana | 3000 | `serve/grafana/dashboards/querycraft.json` (provisioned) |
| jaeger | 16686 | one span per agent step (OTLP on 4318) |

## Option A: one AWS GPU instance

1. `KEY_NAME=... SG_ID=... REGION=us-east-1 bash serve/deploy/aws_g6.sh` (g6.xlarge, one L4 24 GB).
2. Open `http://<ip>:8000`. Restrict the security group to your IP: the API has no authentication.
3. Stop paying: `aws ec2 terminate-instances --instance-ids <id>`.

## Option B: Modal

`modal deploy serve/deploy/modal_serve.py`. Keep the workspace spend limit at $0 so it stops instead of charging.

## Before exposing it to anyone

* Add authentication (an API key header) and rate limiting in front of `/ask`.
* The databases are opened read only, with a 30 s timeout and a 1,000 row cap per query (`querycraft/agent.py`).
* The question log (`runs/live/questions.sqlite`) stores questions verbatim: treat it as user data.
* Image tags in `serve/docker-compose.yml` were not pulled or tested by this project.
