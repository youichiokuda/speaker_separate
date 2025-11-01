FROM python:3.11-slim

# OS deps: ffmpeg + libsndfile for torchaudio soundfile backend
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app after deps to leverage layer cache
COPY app /app/app

ENV DATA_DIR=/data
ENV TRANSFORMERS_CACHE=/data/hf/cache
ENV HF_HOME=/data/hf

# Bind to Render-provided $PORT (fallback 10000 for local)
EXPOSE 10000
CMD ["/bin/sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-10000}"]
