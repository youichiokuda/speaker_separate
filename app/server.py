import os
import uuid
import subprocess
import traceback
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ==== アプリ内部モジュール ====
from app.diarize import run_diarization
from app.transcribe import run_transcription
from app.merge import merge_diarization_and_transcript, write_outputs

# ==========================================================
# FastAPI アプリ設定
# ==========================================================
app = FastAPI(title="Speaker Separation & Transcription API", version="1.3")

# CORS（必要に応じて絞ってOK）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CPUスレッドを絞って安定化（Renderの小メモリ環境向け）
torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))

# 共有データ領域
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# /files で /data を公開（生成物をダウンロード可能に）
app.mount("/files", StaticFiles(directory=str(DATA_DIR)), name="files")


# ==========================================================
# ユーティリティ: 任意フォーマット → 16kHz/mono WAV へ変換
# ==========================================================
def convert_to_wav(src_path: Path, out_dir: Path) -> Path:
    """
    ffmpeg を用いて入力音声を 16kHz/mono の WAV に変換する。
    日本語・スペース等のファイル名も安全に扱うため出力は UUID 名にする。
    """
    dst_path = out_dir / f"{uuid.uuid4().hex}.wav"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src_path),
        "-ac", "1",
        "-ar", "16000",
        str(dst_path),
    ]
    try:
        # 失敗時の stderr を返せるよう PIPE で受ける
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg変換エラー: {e.stderr.decode('utf-8', errors='ignore')}")
    return dst_path


# ==========================================================
# トップページ（簡易フォーム）
# ==========================================================
@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
      <head><title>Speaker Separation</title></head>
      <body style="font-family: sans-serif; max-width: 720px; margin: 32px auto;">
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
        <p style="margin-top:24px;color:#555;">
          完了すると、生成ファイルの <code>/files/...</code> 公開URLが JSON で返ります。
        </p>
      </body>
    </html>
    """


# ==========================================================
# メイン処理エンドポイント
# ==========================================================
@app.post("/api/transcribe")
async def transcribe_api(
    file: UploadFile = File(...),
    whisper_model: str = Form("small"),
    language: str = Form("ja"),
    num_speakers: str = Form("auto"),
):
    """
    アップロードされた音声/動画を
    1) 16kHz/mono WAV へ統一
    2) 話者分離（pyannote）
    3) 文字起こし（faster-whisper）
    4) セグメントをマージして各種フォーマットで出力
    まで実施し、/files で参照可能な公開URLも返す。
    """
    try:
        # --- 一時保存（日本語ファイル名OK） ---
        input_path = DATA_DIR / file.filename
        with open(input_path, "wb") as f:
            f.write(await file.read())

        # --- 必ず WAV に変換して以降は WAV を使用 ---
        print("==> Converting to WAV if necessary...")
        src_path = input_path
        if input_path.suffix.lower() != ".wav":
            src_path = convert_to_wav(input_path, DATA_DIR)
        print(f"Using source file: {src_path}")

        # --- 1. 話者分離 ---
        print("==> 1/3 Diarization...")
        diarization = run_diarization(src_path, num_speakers=num_speakers)

        # --- 2. 文字起こし ---
        print("==> 2/3 Transcription...")
        transcript = run_transcription(src_path, whisper_model, language)

        # --- 3. マージ ---
        print("==> 3/3 Merge...")
        merged_segments = merge_diarization_and_transcript(diarization, transcript)

        # --- 出力（/data/＜元名＞_out/ に作成） ---
        outdir = DATA_DIR / (Path(file.filename).stem + "_out")
        outdir.mkdir(exist_ok=True)
        outputs = write_outputs(merged_segments, outdir)  # {name: Path}

        # --- 公開URL（/files/以下）を作る ---
        public_urls = {}
        for name, path in outputs.items():
            rel = Path(path).relative_to(DATA_DIR)
            public_urls[name] = f"/files/{rel.as_posix()}"

        return {
            "status": "success",
            "message": "Transcription & Diarization complete.",
            "outputs": {k: str(v) for k, v in outputs.items()},  # サーバ上の絶対パス
            "urls": public_urls,                                  # ブラウザでアクセス可能なURL
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
