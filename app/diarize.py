import os
from pathlib import Path
from typing import Union
from huggingface_hub import login
from pyannote.audio import Pipeline


def run_diarization(
    media_path: Union[str, Path],
    num_speakers: Union[int, str] = "auto",
    hf_token: str = None
):
    """
    音声ファイルを話者分離（diarization）して結果を返す。

    Parameters
    ----------
    media_path : str or Path
        分析する音声ファイルへのパス。
    num_speakers : int or 'auto'
        話者数を指定する。'auto' の場合は自動判定。
    hf_token : str, optional
        Hugging Face のアクセストークン。未指定の場合は環境変数 HUGGINGFACE_TOKEN を使用。
    """

    # --- トークン取得 ---
    token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("Hugging Face token is required. Set HUGGINGFACE_TOKEN environment variable.")

    # --- ログイン処理 ---
    login(token=token, add_to_git_credential=False)

    # --- pyannote パイプラインの読み込み ---
    print("Loading Pyannote speaker diarization pipeline...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

    # --- 音声の話者分離実行 ---
    print(f"Running diarization on: {media_path}")
    diarization = pipeline(
        str(media_path),
        num_speakers=None if num_speakers == "auto" else int(num_speakers),
    )

    results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        results.append({
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "speaker": speaker
        })

    print(f"Diarization complete: {len(results)} segments detected.")
    return results


if __name__ == "__main__":
    # テスト用（ローカルで実行したい場合）
    test_audio = "sample.wav"
    if os.path.exists(test_audio):
        res = run_diarization(test_audio)
        for r in res:
            print(r)
    else:
        print("Please place an audio file named 'sample.wav' in the current directory.")
