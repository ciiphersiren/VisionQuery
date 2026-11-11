# VisionQuery: one container with the API, the web interface and the detector weights.
# Runs on CPU. Works locally (docker run) and as a Hugging Face Docker Space (port 7860).
FROM python:3.12-slim

# OpenCV (used by Ultralytics) needs these system libraries
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run the container as user 1000; using the same user locally keeps both identical
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    PORT=7860
WORKDIR /app

# 1) Dependencies first, so code changes do not reinstall them. CPU-only PyTorch keeps the image ~2 GB smaller.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && pip install "ultralytics==8.4.174" "sahi==0.12.8" ensemble-boxes numpy pillow "pydantic>=2" pyyaml \
                fastapi uvicorn python-multipart httpx

# 2) The app itself (code, configs, built interface, sample frames, weights)
COPY --chown=user . /app
RUN pip install --no-deps -e . && chown -R user /app
USER user

# 3) Fetch the zero-shot open-vocabulary model and its text encoder now, so the first "how many trees?" is not slow.
#    Not fatal: if the download fails here, Ultralytics retries on first use.
RUN python -c "from ultralytics import YOLOE; m = YOLOE('yoloe-26s-seg.pt'); m.get_text_pe(['tree'])" || true

EXPOSE 7860
CMD ["sh", "-c", "visionquery serve --host 0.0.0.0 --port ${PORT}"]
