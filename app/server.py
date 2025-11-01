from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uuid, os, sys
from app.main import main as pipeline_main

app = FastAPI(title="Speaker Diarization + ASR")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DATA_DIR = Path(os.getenv("DATA_DIR", "/data")).resolve()
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

HTML_FORM = """<!doctype html><html><body>
<h1>話者分離 + 文字起こし</h1>
<form action='/api/transcribe' method='post' enctype='multipart/form-data'>
<p><input type='file' name='file' accept='.wav,.mp3,.m4a,.mp4' required></p>
<p>Whisperモデル: <input name='whisper_model' value='small'></p>
<p>言語（空で自動）: <input name='language' value='ja'></p>
<p>話者数（auto or 数値）: <input name='num_speakers' value='auto'></p>
<p><button type='submit'>実行</button></p></form></body></html>"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_FORM

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...), whisper_model: str = Form("small"),
                     language: str | None = Form(None), num_speakers: str = Form("auto")):
    uid = uuid.uuid4().hex[:8]
    src_path = UPLOAD_DIR / f"{uid}_{file.filename}"
    with open(src_path, "wb") as f:
        f.write(await file.read())

    outdir = DATA_DIR / "outputs" / uid
    outdir.mkdir(parents=True, exist_ok=True)
    sys.argv = ["main.py", "--input", str(src_path), "--outdir", str(outdir),
                "--whisper_model", whisper_model, "--num_speakers", num_speakers]
    if language:
        sys.argv += ["--language", language]
    try:
        pipeline_main()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    base = f"/api/download/{uid}"
    return {
        "id": uid,
        "files": {
            "markdown": f"{base}/transcript_speaker.md",
            "srt": f"{base}/transcript_speaker.srt",
            "vtt": f"{base}/transcript_speaker.vtt",
            "csv": f"{base}/segments.csv",
        },
    }

@app.get("/api/download/{uid}/{filename}")
def download(uid: str, filename: str):
    fpath = DATA_DIR / "outputs" / uid / filename
    if not fpath.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(fpath)
