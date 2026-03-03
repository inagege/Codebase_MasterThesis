#!/usr/bin/env python3
import argparse
import csv
import shutil
import subprocess
import tempfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.apply_all_audio_noise import (
    AUDIO_CORRUPTIONS,
    apply_audio_corruption,
    extract_audio_only,
)

WAV_EXTS = {".wav"}


def _ffmpeg():
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH")


def _run(cmd: list[str]):
    subprocess.run(cmd, check=True)


def _append_error_row(path: Path, file_name: str, error: str):
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "error"])
        if write_header:
            writer.writeheader()
        writer.writerow({"file": file_name, "error": error})


def _iter_wavs(inp: Path, recursive: bool):
    if inp.is_file():
        if inp.suffix.lower() in WAV_EXTS:
            yield inp
        return
    if recursive:
        for p in inp.rglob("*"):
            if p.is_file() and p.suffix.lower() in WAV_EXTS:
                yield p
    else:
        for p in inp.iterdir():
            if p.is_file() and p.suffix.lower() in WAV_EXTS:
                yield p


def _wav_to_tmp_video(wav_path: Path, tmp_video: Path):
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x16:r=25",
            "-i",
            str(wav_path),
            "-shortest",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "stillimage",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "pcm_s16le",
            str(tmp_video),
        ]
    )


def main():
    ap = argparse.ArgumentParser("Apply all audio perturbations to WAV files.")
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--severity", type=int, default=3)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--audio_sr", type=int, default=16000)
    args = ap.parse_args()

    _ffmpeg()

    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(list(_iter_wavs(audio_dir, args.recursive)))
    if not wavs:
        raise RuntimeError(f"No wav files found in {audio_dir}")

    for corr in AUDIO_CORRUPTIONS:
        combo_root = out_dir / f"A={corr}_S={args.severity}"
        combo_root.mkdir(parents=True, exist_ok=True)
        error_csv = combo_root / "error.csv"

        for wav in wavs:
            rel = wav.relative_to(audio_dir) if audio_dir.is_dir() else Path(wav.name)
            out_wav = (combo_root / rel).with_suffix(".wav")
            out_wav.parent.mkdir(parents=True, exist_ok=True)

            if out_wav.exists() and not args.overwrite:
                continue

            try:
                with tempfile.TemporaryDirectory() as td:
                    td_path = Path(td)
                    src_video = td_path / "src.mp4"
                    out_video = td_path / "corrupted.mp4"
                    _wav_to_tmp_video(wav, src_video)
                    processed_video = apply_audio_corruption(
                        src_video, out_video, corr, args.severity, overwrite=True
                    )
                    extract_audio_only(processed_video, out_wav, overwrite=True, sr=args.audio_sr)
            except Exception as exc:
                _append_error_row(error_csv, str(wav), str(exc))
                print(f"[WARN] Failed {wav}: {exc}")
                continue

        print(f"[OK] Finished audio corruption: {combo_root}")

    print("[DONE] All audio perturbations for WAV input applied.")


if __name__ == "__main__":
    main()
