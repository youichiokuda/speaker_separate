import argparse
from pathlib import Path
from diarize import run_diarization
from transcribe import run_transcription
from merge import merge_diarization_and_transcript, write_outputs

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="audio/video file")
    p.add_argument("--outdir", default="outputs", help="output directory")
    p.add_argument("--whisper_model", default="medium", help="Whisper model size")
    p.add_argument("--language", default=None, help="language code like 'ja'")
    p.add_argument("--num_speakers", default="auto", help="'auto' or integer")
    p.add_argument("--hf_token", default=None, help="Hugging Face token")
    p.add_argument("--vad_filter", action="store_true", help="use VAD filter")
    return p.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("==> 1/3 Diarization...")
    diar_segments = run_diarization(
        media_path=Path(args.input),
        num_speakers=args.num_speakers,
        hf_token=args.hf_token
    )

    print("==> 2/3 Transcription...")
    asr_segments = run_transcription(
        media_path=Path(args.input),
        model_size=args.whisper_model,
        language=args.language,
        vad_filter=args.vad_filter
    )

    print("==> 3/3 Merge & Export...")
    merged, df = merge_diarization_and_transcript(diar_segments, asr_segments)
    write_outputs(merged, df, outdir)
    print(f"✅ Done. Outputs saved in {outdir.resolve()}")

if __name__ == "__main__":
    main()
