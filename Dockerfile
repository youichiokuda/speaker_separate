# ベースイメージ
FROM python:3.11-slim

# ===== OS 依存パッケージ =====
# ffmpeg: 音声変換
# libsndfile1: torchaudio の soundfile バックエンド
# git: HFモデル取得で使う可能性
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git \
 && rm -rf /var/lib/apt/lists/*

# ===== ランタイム環境 =====
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# ===== 作業ディレクトリ =====
WORKDIR /app

# ===== Python 依存ライブラリ =====
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
 && python -m pip install -r requirements.txt

# ===== アプリケーションのコピー =====
COPY app /app/app

# ===== データキャッシュ/書き込み先 =====
ENV DATA_DIR=/data \
    TRANSFORMERS_CACHE=/data/hf/cache \
    HF_HOME=/data/hf
# 初回起動時の書き込みエラー回避（権限ゆるめ）
RUN mkdir -p /data && chmod -R 777 /data

# ===== Render のポートにバインド =====
EXPOSE 10000
CMD ["/bin/sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-10000}"]
