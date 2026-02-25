import argparse
import csv
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=str, required=True, help="Directory containing .mp4 files")
    p.add_argument(
        "--audio-subdir",
        type=str,
        default="audio_only",
        help="Subdirectory name (created under input-dir) to store wav files",
    )
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
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel ffmpeg workers. Use 1 for serial execution.",
    )
    return p.parse_args()


def ffmpeg_extract_wav(mp4_path: Path, wav_path: Path, channels: int, overwrite: bool):
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
        "-threads", "1",        # one ffmpeg thread per worker process
        "-ac", str(channels),   # channels
        "-acodec", "pcm_s32le", # WAV codec
        str(wav_path),
    ]

    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    audio_dir = input_dir / args.audio_subdir
    audio_dir.mkdir(parents=True, exist_ok=True)

    errors_csv = Path(args.errors_csv) if args.errors_csv else (audio_dir / "_errors.csv")

    # find mp4s recursively (will search speaker/id and nested random-letter subfolders)
    mp4s = sorted(input_dir.rglob("*.mp4"))
    print(f"[INFO] Scanning: {input_dir}")
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
    to_process = []

    for i, mp4_path in enumerate(mp4s, start=1):
        # Save WAV with same base name
        # Preserve the mp4's relative path under input_dir so output keeps speaker/random subfolders
        rel = mp4_path.relative_to(input_dir)
        wav_path = audio_dir / rel.with_suffix(".wav")

        if wav_path.exists() and not args.overwrite:
            skipped += 1
            if i == 1 or i % 100 == 0:
                print(f"[INFO] {i}/{len(mp4s)} skipped existing: {wav_path.name}")
            continue

        to_process.append((mp4_path, wav_path))

    print(f"[INFO] workers={args.workers}")
    print(f"[INFO] pending extraction files={len(to_process)}")

    if args.workers == 1:
        for i, (mp4_path, wav_path) in enumerate(to_process, start=1):
            try:
                ffmpeg_extract_wav(mp4_path, wav_path, channels=args.channels, overwrite=args.overwrite)
                ok += 1
                if i == 1 or i % 100 == 0:
                    print(f"[INFO] {i}/{len(to_process)} extracted: {wav_path.name}")
            except Exception as e:
                failed += 1
                err_writer.writerow(
                    {"file": mp4_path.name, "mp4_path": str(mp4_path), "wav_path": str(wav_path), "error": str(e)}
                )
    else:
        chunk_size = 5000
        completed = 0
        for start in range(0, len(to_process), chunk_size):
            chunk = to_process[start : start + chunk_size]
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_to_item = {
                    executor.submit(
                        ffmpeg_extract_wav,
                        mp4_path,
                        wav_path,
                        args.channels,
                        args.overwrite,
                    ): (mp4_path, wav_path)
                    for mp4_path, wav_path in chunk
                }
                for future in as_completed(future_to_item):
                    mp4_path, wav_path = future_to_item[future]
                    completed += 1
                    try:
                        future.result()
                        ok += 1
                        if completed == 1 or completed % 100 == 0:
                            print(f"[INFO] {completed}/{len(to_process)} extracted: {wav_path.name}")
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
