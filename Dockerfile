# Serves the fraud detection API (CPU engine -- see README "REST API" for
# why CPU is the default: it measured fastest at batch_size=1, which is
# what a single /predict call is).
#
# Deliberately lean: only what src/api/server.py + src/inference/cpu_inference.py
# actually import at runtime (torch, fastapi, uvicorn). requirements.txt
# includes pandas/scipy/scikit-learn too, but those are training/preprocessing
# dependencies -- serving doesn't need them, so this image doesn't install them.

FROM python:3.11-slim

WORKDIR /app

# CPU-only torch wheel specifically (the default PyPI torch wheel bundles
# CUDA and is ~10x larger; this image never touches a GPU).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir fastapi "uvicorn[standard]"

# Application code + web demo frontend (served by the API itself, see
# src/api/server.py's StaticFiles mount)
COPY src/ src/
COPY frontend/ frontend/

# Sample transactions for the GET /samples endpoint (used by the frontend's
# "Load Example" buttons) -- without this the endpoint returns an empty
# list and the demo buttons have nothing to load.
COPY data/samples/ data/samples/

# Trained model checkpoint -- NOT committed to git (see .gitignore), so this
# COPY requires `python -m src.models.train` to have been run in the build
# context first. See README "Reproduction" for the full setup sequence.
COPY models/checkpoints/fraud_mlp.pt models/checkpoints/fraud_mlp.pt
COPY models/checkpoints/fraud_mlp_config.json models/checkpoints/fraud_mlp_config.json

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
