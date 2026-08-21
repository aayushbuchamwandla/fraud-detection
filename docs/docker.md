# Phase 10: Docker

## Environment resolution

Flagged from the project's start as the one checkpoint most likely to need manual admin action (Docker Desktop's installer is genuinely admin-gated on Windows, no way around that specific installer). **Resolved with zero manual input** by installing Docker Engine directly inside WSL2 via `apt` — not Docker Desktop for Windows at all:

```bash
sudo apt-get install -y docker.io
sudo service docker start   # sysvinit-style start; WSL2 doesn't run systemd by default, and this doesn't need it
```

This works because WSL2 is a real Linux kernel, not a compatibility layer — `docker.io`'s daemon runs the same way it would on any Ubuntu server, no Docker Desktop, no Hyper-V/WSL2-backend GUI dance, no reboot. The passwordless `sudo` set up back in Phase 6 for the CUDA toolchain covered this too, so every command here ran unattended.

`docker-compose` (v1, from `apt install docker-compose`) also works the same way — `docker-compose-plugin` (v2) isn't in Ubuntu 22.04's default apt sources, so v1 was used instead. Functionally equivalent for this project's single-service `docker-compose.yml`.

## A real bug this testing caught

The first build/run of the image (before the web frontend was added) worked correctly. After adding the frontend and its `GET /samples` endpoint, actually testing the container (not just rebuilding and assuming) showed `/samples` returning `[]` — the Dockerfile copied `src/` and `frontend/` but not `data/samples/`, so the sample-transaction CSV simply wasn't in the image. Fixed by adding `COPY data/samples/ data/samples/`, rebuilt, and reverified the endpoint returns the real 10 labeled transactions. Left in this doc as a concrete example of why every phase in this project gets actually run, not just written and assumed correct.

## What's containerized

Only the REST API (Phase 9), serving with the **CPU engine** — consistent with that phase's own data-driven default (CPU measured fastest at batch_size=1). The `Dockerfile` is deliberately lean: it installs only what `src/api/server.py` and `src/inference/cpu_inference.py` actually import at runtime (`torch` CPU wheel, `fastapi`, `uvicorn`), not the full `requirements.txt` (which includes `pandas`/`scipy`/`scikit-learn` for training/preprocessing — irrelevant to serving).

The CUDA/TensorRT/custom-kernel/C++ backends are **not** containerized. That's a real, disclosed limitation, not an oversight: GPU passthrough into a WSL2-hosted Docker Engine is its own separate configuration surface (NVIDIA Container Toolkit, `--gpus` runtime flags) that wasn't set up or tested here. `FRAUD_API_ENGINE` supports selecting them, but doing so inside this specific container would fail without that additional GPU-passthrough setup — documented as unverified rather than silently implied to work.

## Verification (real, not assumed)

```
$ docker build -t fraud-api:latest .
Successfully built 97c181f4a7ef
Successfully tagged fraud-api:latest

$ docker run -d --name fraud-api-test -p 8000:8000 fraud-api:latest
$ docker ps
CONTAINER ID   ...   STATUS                   PORTS
652a1c8561c6   ...   Up 3 seconds (healthy)   0.0.0.0:8000->8000/tcp

$ curl http://localhost:8000/health
{"status":"ok","engine":"cpu","model_loaded":true,"decision_threshold":0.9992}

# Real fraud transaction (test-split index 31474):
{"fraud_probability": 1.0, "is_fraud": true, "engine": "cpu", "latency_ms": 1.62}

# Real legitimate transaction (test-split index 40337):
{"fraud_probability": 1.04e-07, "is_fraud": false, "engine": "cpu", "latency_ms": 0.29}
```

Identical results to the non-containerized live-server test in Phase 9 — same model, same weights, same predictions, just running inside a container instead of directly on the host. `docker-compose up -d` was verified separately and produced the same healthy, correctly-serving container.

## Reproduction

```bash
# Inside WSL2 (or any Linux Docker host):
sudo apt-get install -y docker.io docker-compose
sudo service docker start

# From the project root (requires a trained checkpoint -- see README Reproduction):
docker build -t fraud-api:latest .
docker run -d -p 8000:8000 fraud-api:latest
curl http://localhost:8000/health

# or:
docker-compose up -d
docker-compose down
```
