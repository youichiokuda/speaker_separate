import os
import uuid
import time
import subprocess
import traceback
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

# ==== アプリ内部モジュール ====
from app.diarize import run_diarization
from app.transcribe import run_transcription
from app.merge import merge_diarization_and_transcript, write_outputs


# ==========================================================
# FastAPI 設定
# ==========================================================
app = FastAPI(title="Speaker Separation & Transcription API", version="1.4")

# CORS（Renderなどからの呼び出しを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CPUスレッド制御（Renderで安定稼働）
torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))

# データディレクトリ設定
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 静的ファイルとして公開（/files/... でアクセス可）
app.mount("/files", StaticFiles(directory=str(DATA_DIR)), name="files")


# ==========================================================
# ミドルウェア: アップロードサイズ制限
# ==========================================================
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl and int(cl) > 200 * 1024 * 1024:  # 200MB上限
        return PlainTextResponse("File too large", status_code=413)
    return await call_next(request)


# ==========================================================
# ユーティリティ関数
# ==========================================================
def convert_to_wav(src_path: Path, out_dir: Path) -> Path:
    """ffmpegで16kHz/mono WAVへ変換"""
    dst_path = out_dir / f"{uuid.uuid4().hex}.wav"
    cmd = ["ffmpeg", "-y", "-i", str(src_path), "-ac", "1", "-ar", "16000", str(dst_path)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg変換エラー: {e.stderr.decode('utf-8', errors='ignore')}")
    return dst_path


def probe_duration(path: Path) -> float:
    """ffprobeで音声の長さを秒単位で返す"""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return float(out.decode().strip())
    except Exception:
        return 0.0


# ==========================================================
# トップページ（フォームUI）
# ==========================================================
@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
      <head><title>Speaker Separation</title></head>
      <body style="font-family:sans-serif;max-width:720px;margin:32px auto;">
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
          完了すると /files/... でアクセスできる出力URLが返ります。
        </p>
      </body>
    </html>
    """


# ==========================================================
# メインエンドポイント
# ==========================================================
@app.post("/api/transcribe")
async def transcribe_api(
    file: UploadFile = File(...),
    whisper_model: str = Form("small"),
    language: str = Form("ja"),
    num_speakers: str = Form("auto"),
    request: Request = None,
):
    rid = uuid.uuid4().hex[:8]  # リクエストID
    try:
        print(f"[{rid}] ==== New Request ====")
        input_path = DATA_DIR / file.filename
        with open(input_path, "wb") as f:
            f.write(await file.read())

        # 変換
        print(f"[{rid}] Converting to WAV if necessary...")
        src_path = input_path
        if input_path.suffix.lower() != ".wav":
            src_path = convert_to_wav(input_path, DATA_DIR)
        print(f"[{rid}] Source file: {src_path}")

        # 長さチェック
        dur = probe_duration(src_path)
        print(f"[{rid}] Duration: {dur:.1f}s")
        if dur > 5400:  # 90分上限
            return JSONResponse({"error": f"Audio too long: {dur:.1f}s > 5400s"}, status_code=400)

        # 1. Diarization
        print(f"[{rid}] ==> 1/3 Diarization start...")
        t0 = time.time()
        diarization = run_diarization(src_path, num_speakers=num_speakers)
        print(f"[{rid}] Diarization done ({time.time()-t0:.1f}s)")

        # 2. Transcription
        print(f"[{rid}] ==> 2/3 Transcription start...")
        t1 = time.time()
        transcript = run_transcription(src_path, whisper_model, language)
        print(f"[{rid}] Transcription done ({time.time()-t1:.1f}s)")

        # 3. Merge
        print(f"[{rid}] ==> 3/3 Merge start...")
        t2 = time.time()
        merged_segments = merge_diarization_and_transcript(diarization, transcript)
        print(f"[{rid}] Merge done ({time.time()-t2:.1f}s)")

        # 出力保存
        outdir = DATA_DIR / f"{uuid.uuid4().hex}_out"
        outdir.mkdir(exist_ok=True)
        outputs = write_outputs(merged_segments, outdir)

        # 公開URL（絶対URL形式）
        base = str(request.base_url).rstrip("/")
        public_urls = {
            name: f"{base}/files/{Path(path).relative_to(DATA_DIR).as_posix()}"
            for name, path in outputs.items()
        }

        print(f"[{rid}] ==== Completed ====")
        return {
            "request_id": rid,
            "status": "success",
            "message": "Transcription & Diarization complete.",
            "outputs": {k: str(v) for k, v in outputs.items()},
            "urls": public_urls,
        }

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[{rid}] ERROR:\n{tb}")
        return JSONResponse({"error": str(e), "traceback": tb, "request_id": rid}, status_code=500)


# ==========================================================
# ヘルスチェック
# ==========================================================
@app.get("/healthz")
def health_check():
    return {"status": "ok"}


# ==========================================================
# ウォームアップ（モデル事前読み込みに利用可）
# ==========================================================
@app.get("/warmup")
def warmup():
    return {"status": "ready"}
