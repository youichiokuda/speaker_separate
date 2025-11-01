# ベースイメージ
FROM python:3.11-slim

# ===== 依存パッケージのインストール =====
# ffmpeg: 音声変換に必要
# libsndfile1: torchaudio の soundfile バックエンド用
# git: Hugging Face モデルのクローンで使用される場合がある
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git \
 && rm -rf /var/lib/apt/lists/*

# ===== 作業ディレクトリ =====
WORKDIR /app

# ===== Python依存ライブラリのインストール =====
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# ===== アプリケーションのコピー =====
COPY app /app/app

# ===== キャッシュ・環境変数設定 =====
ENV DATA_DIR=/data
ENV TRANSFORMERS_CACHE=/data/hf/cache
ENV HF_HOME=/data/hf

# ===== Renderが提供するポートにバインド =====
# RenderはPORTという環境変数を自動で設定するため、
# これを利用してポートを明示的にバインドする
EXPOSE 10000
CMD ["/bin/sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-10000}"]
