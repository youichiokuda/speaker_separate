import os
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
from pyannote.audio import Pipeline

@dataclass
class DiarSegment:
    start: float
    end: float
    speaker: str

def run_diarization(media_path: Path, num_speakers: str = "auto",
                    hf_token: Optional[str] = None) -> List[DiarSegment]:
    token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("Hugging Face token is required.")

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    diarization = pipeline(str(media_path), num_speakers=None if num_speakers == "auto" else int(num_speakers))

    segments: List[DiarSegment] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(DiarSegment(start=turn.start, end=turn.end, speaker=str(speaker)))

    mapping, counter = {}, 0
    for s in segments:
        if s.speaker not in mapping:
            mapping[s.speaker] = f"SPEAKER_{counter:02d}"
            counter += 1
        s.speaker = mapping[s.speaker]
    return segments
