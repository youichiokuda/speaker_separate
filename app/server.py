import os
import traceback
import subprocess
import uuid
from pathlib import Path
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# ====== アプリ内部モジュール ======
from app.diarize import run_diarization
from app.transcribe import run_transcription
from app.merge import merge_diarization_and_transcript, write_outputs

# ==========================================================
# FastAPI アプリ設定
# ==========================================================
app = FastAPI(title="Speaker Separation & Transcription API", version="1.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path("/data")
DATA_DIR.mkdir(exist_ok=True)

# ==========================================================
# ffmpeg: m4a / mp4 などを WAV に変換
# ==========================================================
def convert_to_wav(src_path: Path, out_dir: Path) -> Path:
    """任意フォーマットを16kHz mono WAVに変換"""
    dst_path = out_dir / f"{uuid.uuid4().hex}.wav"
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(src_path),
            "-ac", "1", "-ar", "16000",
            str(dst_path)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg変換エラー: {e.stderr.decode('utf-8', errors='ignore')}")
    return dst_path


# ==========================================================
# トップページ（フォームUI）
# ==========================================================
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


# ==========================================================
# /api/transcribe: メイン処理
# ==========================================================
@app.post("/api/transcribe")
async def transcribe_api(
    file: UploadFile = File(...),
    whisper_model: str = Form("small"),
    language: str = Form("ja"),
    num_speakers: str = Form("auto"),
):
    """音声ファイルを受け取り、話者分離＋文字起こし＋マージを実施"""
    try:
        # --- 一時ファイル保存 ---
        input_path = DATA_DIR / file.filename
        with open(input_path, "wb") as f:
            f.write(await file.read())

        # --- m4a/mp4等をWAVへ変換 ---
        print("==> Converting to WAV if necessary...")
        src = input_path
        if input_path.suffix.lower() != ".wav":
            src = convert_to_wav(input_path, DATA_DIR)
        print(f"Using source file: {src}")

        # --- 1. 話者分離 ---
        print("==> 1/3 Diarization...")
        diarization = run_diarization(src, num_speakers=num_speakers)

        # --- 2. 文字起こし ---
        print("==> 2/3 Transcription...")
        transcript = run_transcription(src, whisper_model, language)

        # --- 3. マージ ---
        print("==> 3/3 Merge...")
        merged_segments = merge_diarization_and_transcript(diarization, transcript)

        # --- 出力生成 ---
        outdir = DATA_DIR / (Path(file.filename).stem + "_out")
        outdir.mkdir(exist_ok=True)
        outputs = write_outputs(merged_segments, outdir)

        return {
            "status": "success",
            "message": "Transcription & Diarization complete.",
            "outputs": {k: str(v) for k, v in outputs.items()},
        }

    except Exception as e:
        tb = traceback.format_exc()
        print("[/api/transcribe] ERROR\n", tb)
        return JSONResponse({"error": str(e), "traceback": tb}, status_code=500)


# ==========================================================
# Render ヘルスチェック
# ==========================================================
@app.get("/healthz")
def health_check():
    return {"status": "ok"}
