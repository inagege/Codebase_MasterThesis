import argparse
import csv
import os
import subprocess
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=str, required=True, help="Directory containing .mp4 files")
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Search for mp4 files recursively with rglob (default: only top-level)",
    )
    p.add_argument(
        "--audio-subdir",
        type=str,
        default="audio_only",
        help="Subdirectory name (created under input-dir) to store wav files",
    )
    p.add_argument("--sr", type=int, default=16000, help="Output sample rate")
    p.add_argument("--channels", type=int, default=1, help="Number of audio channels (1=mono, 2=stereo)")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing wav files",
    )
    p.add_argument(
        "--errors-csv",
        type=str,
        default=None,
        help="Optional path to write errors CSV. Default: <input-dir>/<audio-subdir>/_errors.csv",
    )
    return p.parse_args()


def ffmpeg_extract_wav(mp4_path: Path, wav_path: Path, sr: int, channels: int, overwrite: bool):
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if overwrite:
        cmd += ["-y"]
    else:
        cmd += ["-n"]

    # Input
    cmd += ["-i", str(mp4_path)]

    # Convert: PCM 16-bit WAV, resample + set channels
    cmd += [
        "-vn",                  # no video
        "-ac", str(channels),   # channels
        "-ar", str(sr),         # sample rate
        "-acodec", "pcm_s16le", # WAV codec
        str(wav_path),
    ]

    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    audio_dir = input_dir / args.audio_subdir
    audio_dir.mkdir(parents=True, exist_ok=True)

    errors_csv = Path(args.errors_csv) if args.errors_csv else (audio_dir / "_errors.csv")

    mp4s = sorted(input_dir.rglob("*.mp4")) if args.recursive else sorted(input_dir.glob("*.mp4"))
    print(f"[INFO] Scanning: {input_dir} (recursive={args.recursive})")
    print(f"[INFO] Found {len(mp4s)} mp4 files")
    print(f"[INFO] Writing wav to: {audio_dir}")

    # Prepare error CSV
    write_header = not errors_csv.exists()
    err_f = open(errors_csv, "a", newline="", encoding="utf-8")
    err_writer = csv.DictWriter(err_f, fieldnames=["file", "mp4_path", "wav_path", "error"])
    if write_header:
        err_writer.writeheader()

    ok = 0
    skipped = 0
    failed = 0

    for i, mp4_path in enumerate(mp4s, start=1):
        # Save WAV with same base name
        wav_path = audio_dir / (mp4_path.stem + ".wav")

        if wav_path.exists() and not args.overwrite:
            skipped += 1
            if i == 1 or i % 100 == 0:
                print(f"[INFO] {i}/{len(mp4s)} skipped existing: {wav_path.name}")
            continue

        try:
            ffmpeg_extract_wav(mp4_path, wav_path, sr=args.sr, channels=args.channels, overwrite=args.overwrite)
            ok += 1
            if i == 1 or i % 100 == 0:
                print(f"[INFO] {i}/{len(mp4s)} extracted: {wav_path.name}")
        except Exception as e:
            failed += 1
            err_writer.writerow(
                {"file": mp4_path.name, "mp4_path": str(mp4_path), "wav_path": str(wav_path), "error": str(e)}
            )

    err_f.close()

    print(f"[DONE] ok={ok} skipped={skipped} failed={failed}")
    print(f"[DONE] errors_csv={errors_csv}")


if __name__ == "__main__":
    main()
