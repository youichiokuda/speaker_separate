from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
from faster_whisper import WhisperModel

@dataclass
class AsrSegment:
    start: float
    end: float
    text: str

def run_transcription(media_path: Path, model_size: str = "medium",
                      language: Optional[str] = None, vad_filter: bool = False) -> List[AsrSegment]:
    model = WhisperModel(model_size, device="cuda" if _has_cuda() else "cpu", compute_type=_compute_type())
    segments, _ = model.transcribe(str(media_path), language=language, vad_filter=vad_filter, beam_size=5)
    return [AsrSegment(float(seg.start), float(seg.end), seg.text.strip()) for seg in segments]

def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def _compute_type():
    return "float16" if _has_cuda() else "int8"
