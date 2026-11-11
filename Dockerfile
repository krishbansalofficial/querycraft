# Same stack as the laptop (torch 2.13 cu129 + vLLM 0.32 cu129), for the rented A100 or any CUDA 12.x host.
#   docker build -t querycraft .
#   docker run --gpus all -v $PWD/runs:/app/runs -v $PWD/data:/app/data querycraft make all PROFILE=rented
FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PATH=/root/.local/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-dev gcc make git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* && curl -LsSf https://astral.sh/uv/install.sh | sh
WORKDIR /app
COPY requirements.txt scripts/setup_env.sh scripts/
RUN mv scripts/requirements.txt . && bash scripts/setup_env.sh
COPY . .
CMD ["make", "test"]
