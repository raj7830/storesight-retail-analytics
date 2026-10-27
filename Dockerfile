# CPU image (portable). For an NVIDIA GPU, swap the torch index URL for the
# matching CUDA wheel and run with `--gpus all`; device "auto" picks it up.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface YOLO_CONFIG_DIR=/app/.cache/ultralytics

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY configs/ configs/
COPY models/ models/
COPY src/ src/

# Bake all public weights into the image so inference runs fully offline:
# YOLO11 pose + detector (Ultralytics release assets) and CLIP ViT-B/32 (LAION).
RUN python -m src.tools.download_models

EXPOSE 8000
# Default: the web app. Override the command to use the CLI, e.g.
#   docker run -v $PWD/Input_Video:/app/Input_Video -v $PWD/outputs:/app/outputs retail-analytics \
#       python -m src.run Input_Video/sample1.mp4 Input_Video/sample3.mp4
CMD ["uvicorn", "src.web.server:app", "--host", "0.0.0.0", "--port", "8000"]
