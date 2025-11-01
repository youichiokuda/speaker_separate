from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd
import math

@dataclass
class MergedSegment:
    start: float
    end: float
    speaker: str
    text: str

def merge_diarization_and_transcript(diar_segments, asr_segments) -> Tuple[List[MergedSegment], pd.DataFrame]:
    merged: List[MergedSegment] = []
    for a in asr_segments:
        mid = (a.start + a.end) / 2
        spk = _find_speaker(mid, diar_segments)
        merged.append(MergedSegment(a.start, a.end, spk, a.text))
    merged = _coalesce(merged)
    df = pd.DataFrame([{"start": _fmt(s.start), "end": _fmt(s.end), "speaker": s.speaker, "text": s.text} for s in merged])
    return merged, df

def _find_speaker(mid: float, diar_segments):
    for d in diar_segments:
        if d.start <= mid <= d.end:
            return d.speaker
    return "SPEAKER_00"

def _coalesce(segments: List[MergedSegment], max_gap: float = 0.8):
    if not segments: return []
    out = [segments[0]]
    for s in segments[1:]:
        last = out[-1]
        if s.speaker == last.speaker and s.start - last.end <= max_gap:
            out[-1] = MergedSegment(last.start, s.end, s.speaker, last.text + " " + s.text)
        else:
            out.append(s)
    return out

def _fmt(t: float):
    ms = int(round((t - math.floor(t)) * 1000))
    total = int(t)
    h, m, s = total // 3600, (total % 3600) // 60, total % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def write_outputs(merged, df, outdir):
    md = "\n".join([f"**{m.speaker} [{_fmt(m.start)}–{_fmt(m.end)}]**  \n{m.text}\n" for m in merged])
    (outdir / "transcript_speaker.md").write_text(md, encoding="utf-8")
    df.to_csv(outdir / "segments.csv", index=False, encoding="utf-8")
