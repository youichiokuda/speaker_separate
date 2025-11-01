FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY app /app/app
ENV DATA_DIR=/data
ENV TRANSFORMERS_CACHE=/data/hf/cache
ENV HF_HOME=/data/hf
ENV PYTORCH_ENABLE_MPS_FALLBACK=1
EXPOSE 10000
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "10000"]
