import os
import traceback
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from app.diarize import run_diarization
from app.transcribe import run_transcription
from app.merge import merge_diarization_and_transcript, write_outputs

# ======================================
# FastAPI 初期設定
# ======================================
app = FastAPI(title="Speaker Separation & Transcription API", version="1.0")

# CORS（Renderデプロイ後の外部アクセス対応）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 保存ディレクトリ
DATA_DIR = Path("/data")
DATA_DIR.mkdir(exist_ok=True)


# ======================================
# トップページ (HTML)
# ======================================
@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
      <head><title>Speaker Separation</title></head>
      <body style="font-family: sans-serif;">
        <h2>🎙️ Speaker Separation + Transcription</h2>
        <form action="/api/transcribe" method="post" enctype="multipart/form-data">
          <p><input type="file" name="file" accept="audio/*,video/*" required></p>
          <p>Whisper model:
             <select name="whisper_model">
               <option value="small">small</option>
               <option value="medium">medium</option>
               <option value="large">large</option>
             </select>
          </p>
          <p>Language: <input type="text" name="language" value="ja"></p>
          <p>Number of speakers: <input type="text" name="num_speakers" value="auto"></p>
          <p><input type="submit" value="Start"></p>
        </form>
      </body>
    </html>
    """


# ======================================
# メイン処理エンドポイント
# ======================================
@app.post("/api/transcribe")
async def transcribe_api(
    file: UploadFile = File(...),
    whisper_model: str = Form("small"),
    language: str = Form("ja"),
    num_speakers: str = Form("auto"),
):
    """
    音声ファイルを受け取り、話者分離＋文字起こし＋マージを行うAPI
    """

    try:
        # --- 入力ファイルの保存 ---
        input_path = DATA_DIR / file.filename
        with open(input_path, "wb") as f:
            f.write(await file.read())

        # --- 1. 話者分離 ---
        print("==> 1/3 Diarization...")
        diarization = run_diarization(input_path, num_speakers=num_speakers)

        # --- 2. 音声文字起こし ---
        print("==> 2/3 Transcription...")
        transcript = run_transcription(input_path, whisper_model, language)

        # --- 3. 話者情報と文字起こしのマージ ---
        print("==> 3/3 Merge...")
        merged_segments = merge_diarization_and_transcript(diarization, transcript)

        # --- 出力ファイル生成 ---
        output_dir = DATA_DIR / file.filename.replace(".", "_out.")
        output_dir.mkdir(exist_ok=True)
        output_files = write_outputs(merged_segments, output_dir)

        return {
            "status": "success",
            "message": "Transcription and diarization complete.",
            "outputs": {name: str(path) for name, path in output_files.items()},
        }

    except Exception as e:
        # --- エラー時に詳細を返す ---
        tb = traceback.format_exc()
        print("[/api/transcribe] ERROR\n", tb)
        return JSONResponse(
            {"error": str(e), "traceback": tb},
            status_code=500
        )


# ======================================
# Render 健康チェック用
# ======================================
@app.get("/healthz")
def health_check():
    return {"status": "ok"}
