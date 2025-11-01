# 🎙️ Speaker Diarization + ASR on Render (Whisper + Pyannote)

このリポジトリは、**話者分離（スピーカーダイアライゼーション）＋文字起こし**を行う FastAPI ベースの Web アプリです。  
GitHub 経由で [Render](https://render.com) にそのままデプロイできます。

---

## 🚀 機能概要

- **Whisper (faster-whisper)** による高精度文字起こし（日本語対応）  
- **Pyannote.audio** による話者分離（話者の自動識別）  
- **FastAPI** によるシンプルな Web UI & API  
- **出力形式**
  - Markdown（話者ラベル付き全文）
  - SRT / VTT（字幕形式）
  - CSV（セグメント単位の時刻＋話者＋テキスト）  
- **Render デプロイ対応**
  - Dockerfile + render.yaml 付属  
  - /data ディスクを自動マウントし、モデルキャッシュ永続化  
  - HUGGINGFACE_TOKEN による Pyannote 認証対応

---

## 📦 セットアップ（ローカル）

```bash
git clone https://github.com/yourname/speaker-asr-on-render.git
cd speaker-asr-on-render

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# 環境変数（Hugging Face トークンを取得して設定）
export HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx

# 実行
uvicorn app.server:app --host 0.0.0.0 --port 10000
```

---

## ☁️ Render にデプロイする

1. このリポジトリを GitHub に Push  
2. Render にログイン → 「**New → Web Service**」を選択  
3. 対象リポジトリを選び、`render.yaml` が自動検出されるのを確認  
4. 環境変数を設定：  
   - `HUGGINGFACE_TOKEN`: Hugging Face で発行（[https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)）
5. デプロイ完了後、 `/` にアクセスしてフォームから音声をアップロード！

---

## 🧠 API 仕様

### POST `/api/transcribe`

**パラメータ**：  
- `file`: 音声ファイル（mp3 / wav / m4a / mp4）  
- `whisper_model`: Whisperモデル (`tiny`, `base`, `small`, `medium`, `large-v3`)  
- `language`: 言語コード（例：`ja`）  
- `num_speakers`: 話者数 (`auto` または整数)

**レスポンス例**：

```json
{
  "id": "a1b2c3d4",
  "files": {
    "markdown": "/api/download/a1b2c3d4/transcript_speaker.md",
    "srt": "/api/download/a1b2c3d4/transcript_speaker.srt",
    "vtt": "/api/download/a1b2c3d4/transcript_speaker.vtt",
    "csv": "/api/download/a1b2c3d4/segments.csv"
  }
}
```

---

## 📂 出力サンプル

| speaker | start | end | text |
|----------|--------|------|------|
| SPEAKER_00 | 00:00:01.000 | 00:00:05.200 | おはようございます。 |
| SPEAKER_01 | 00:00:05.300 | 00:00:09.000 | よろしくお願いします。 |

---

## ⚙️ 技術スタック

- **Python 3.11**
- **FastAPI + Uvicorn**
- **faster-whisper**
- **pyannote.audio**
- **Docker / Render デプロイ対応**

---

## 💾 環境変数（Render 用）

| 変数名 | 内容 |
|--------|------|
| `HUGGINGFACE_TOKEN` | Hugging Face のアクセストークン（必須） |
| `DATA_DIR` | モデルキャッシュと出力の保存先（`/data` にマウント） |
| `TRANSFORMERS_CACHE` | Hugging Face モデルキャッシュパス |
| `HF_HOME` | モデル設定キャッシュパス |

---

## 🧩 今後の拡張アイデア

- Streamlit ベースのビジュアル化 UI  
- 音声分割（長時間ファイル対応）  
- Kintone・Teams 連携による議事録自動登録  
- Azure Speech / Google Speech との比較モード

---

© 2025 Yoichi Okuda  
Licensed under MIT
