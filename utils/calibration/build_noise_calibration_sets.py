#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.noise.apply_all_audio_noise as audio_noise
from utils.noise.apply_all_image_noise import IMAGE_CORRUPTIONS, apply_image_corruption
from utils.noise.apply_all_text_noise import TEXT_CORRUPTIONS, perturb


SUPPORTED_MODALITIES = {"audio", "image", "text"}
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TEXT_EXTENSIONS_DEFAULT = ".txt,.raw,.md,.tokens"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build severity-balanced noisy calibration sets for audio/image/text. "
            "This creates exactly N clean and per-severity noisy samples per modality."
        )
    )
    parser.add_argument(
        "--out-root",
        type=str,
        required=True,
        help="Output root for generated modality sets.",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default="audio,image,text",
        help="Comma-separated list from audio,image,text.",
    )
    parser.add_argument(
        "--audio-source-dir",
        type=str,
        default="",
        help="Source directory for clean audio files.",
    )
    parser.add_argument(
        "--image-source-dir",
        type=str,
        default="",
        help="Source directory for clean image files.",
    )
    parser.add_argument(
        "--text-source-dir",
        type=str,
        default="",
        help="Source directory for clean text files.",
    )
    parser.add_argument(
        "--clean-count",
        type=int,
        default=500_000,
        help="Number of clean samples per modality.",
    )
    parser.add_argument("--severity-1-count", type=int, default=150_000)
    parser.add_argument("--severity-2-count", type=int, default=100_000)
    parser.add_argument("--severity-3-count", type=int, default=100_000)
    parser.add_argument("--severity-4-count", type=int, default=100_000)
    parser.add_argument("--severity-5-count", type=int, default=50_000)
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Base random seed.",
    )
    parser.add_argument(
        "--clean-link-mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="How to write clean samples (symlink or copy).",
    )
    parser.add_argument(
        "--audio-sr",
        type=int,
        default=16000,
        help="Target audio sample rate for generated noisy WAV files.",
    )
    parser.add_argument(
        "--text-extensions",
        type=str,
        default=TEXT_EXTENSIONS_DEFAULT,
        help="Comma-separated source text file extensions.",
    )
    parser.add_argument(
        "--text-min-chars",
        type=int,
        default=160,
        help="Minimum chars per sampled text chunk.",
    )
    parser.add_argument(
        "--text-max-chars",
        type=int,
        default=600,
        help="Maximum chars per sampled text chunk.",
    )
    parser.add_argument(
        "--text-pool-max-chunks",
        type=int,
        default=2_000_000,
        help="Cap on in-memory source text chunk pool.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite generated outputs if they already exist.",
    )
    return parser.parse_args()


def _parse_modalities(raw: str) -> list[str]:
    tokens = [token.strip().lower() for token in raw.split(",") if token.strip()]
    if not tokens:
        raise ValueError("No modalities provided.")
    unknown = sorted(set(tokens) - SUPPORTED_MODALITIES)
    if unknown:
        raise ValueError(f"Unsupported modalities: {unknown}. Supported: {sorted(SUPPORTED_MODALITIES)}")
    ordered: list[str] = []
    for token in tokens:
        if token not in ordered:
            ordered.append(token)
    return ordered


def _parse_extensions(raw: str) -> set[str]:
    out = set()
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if not token.startswith("."):
            token = f".{token}"
        out.add(token)
    if not out:
        raise ValueError("No text extensions provided.")
    return out


def _iter_files(root: Path, extensions: set[str]):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("._"):
            continue
        if path.suffix.lower() in extensions:
            yield path


def _iter_text_chunks(text: str, min_chars: int, max_chars: int):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return
    buffer: list[str] = []
    cur = 0
    for line in lines:
        line_len = len(line)
        next_len = cur + (1 if cur > 0 else 0) + line_len
        if buffer and next_len > max_chars and cur >= min_chars:
            yield " ".join(buffer)
            buffer = [line]
            cur = line_len
            continue
        buffer.append(line)
        cur = next_len
    if buffer and cur >= min_chars:
        yield " ".join(buffer)


def _allocate_across_corruptions(total: int, corruptions: list[str]) -> dict[str, int]:
    base = total // len(corruptions)
    rem = total % len(corruptions)
    counts = {}
    for idx, corr in enumerate(corruptions):
        counts[corr] = base + (1 if idx < rem else 0)
    return counts


def _safe_unlink(path: Path):
    if path.exists() or path.is_symlink():
        path.unlink()


def _materialize_clean_file(src: Path, dst: Path, *, mode: str, overwrite: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        _safe_unlink(dst)
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
        return
    shutil.copy2(src, dst)


def _run(cmd: list[str]):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# Keep this log suppression local to this high-volume generator.
audio_noise._run = _run
AUDIO_CORRUPTIONS = audio_noise.AUDIO_CORRUPTIONS
apply_audio_corruption = audio_noise.apply_audio_corruption
extract_audio_only = audio_noise.extract_audio_only


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


def _audio_corrupt_to_wav(src_audio: Path, dst_wav: Path, corruption: str, severity: int, audio_sr: int):
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="calib_audio_noise_") as td:
        td_path = Path(td)
        src_wav = td_path / "src.wav"
        _run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(src_audio),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(audio_sr),
                "-c:a",
                "pcm_s16le",
                str(src_wav),
            ]
        )
        src_video = td_path / "src.mp4"
        tmp_out_video = td_path / "corrupted.mp4"
        _wav_to_tmp_video(src_wav, src_video)
        processed = apply_audio_corruption(src_video, tmp_out_video, corruption, severity, overwrite=True)
        extract_audio_only(processed, dst_wav, overwrite=True, sr=audio_sr)


def _write_metadata_header(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "modality",
                "subset",
                "severity",
                "corruption",
                "source_path",
                "output_path",
                "status",
                "error",
            ],
        )
        writer.writeheader()


def _append_metadata_row(path: Path, row: dict[str, str]):
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "modality",
                "subset",
                "severity",
                "corruption",
                "source_path",
                "output_path",
                "status",
                "error",
            ],
        )
        writer.writerow(row)


def _build_audio_set(
    source_dir: Path,
    out_root: Path,
    rng: random.Random,
    clean_count: int,
    severity_counts: dict[int, int],
    clean_mode: str,
    overwrite: bool,
    audio_sr: int,
):
    audio_files = list(_iter_files(source_dir, AUDIO_EXTENSIONS))
    if not audio_files:
        raise RuntimeError(f"No source audio files found in {source_dir}")

    modality_root = out_root / "audio_1m"
    clean_root = modality_root / "clean"
    metadata_path = modality_root / "metadata.csv"
    _write_metadata_header(metadata_path)

    print(f"[INFO] audio source files={len(audio_files)}", flush=True)

    for idx in range(clean_count):
        src = rng.choice(audio_files)
        dst = clean_root / f"sample_{idx:07d}{src.suffix.lower()}"
        try:
            _materialize_clean_file(src, dst, mode=clean_mode, overwrite=overwrite)
            status = "ok"
            error = ""
        except Exception as exc:
            status = "error"
            error = str(exc)
        _append_metadata_row(
            metadata_path,
            {
                "sample_id": f"clean_{idx:07d}",
                "modality": "audio",
                "subset": "clean",
                "severity": "0",
                "corruption": "",
                "source_path": str(src),
                "output_path": str(dst),
                "status": status,
                "error": error,
            },
        )
        if idx % 5000 == 0:
            print(f"[INFO] audio clean progress {idx}/{clean_count}", flush=True)

    sample_offset = 0
    for severity in (1, 2, 3, 4, 5):
        target = severity_counts[severity]
        per_corr = _allocate_across_corruptions(target, AUDIO_CORRUPTIONS)
        for corr in AUDIO_CORRUPTIONS:
            corr_root = modality_root / f"A={corr}_S={severity}"
            corr_target = per_corr[corr]
            for idx in range(corr_target):
                src = rng.choice(audio_files)
                dst = corr_root / f"sample_{sample_offset:07d}.wav"
                sample_offset += 1
                if dst.exists() and not overwrite:
                    status = "skipped_exists"
                    error = ""
                else:
                    try:
                        _audio_corrupt_to_wav(src, dst, corr, severity, audio_sr)
                        status = "ok"
                        error = ""
                    except Exception as exc:
                        status = "error"
                        error = str(exc)
                _append_metadata_row(
                    metadata_path,
                    {
                        "sample_id": f"noise_{sample_offset:07d}",
                        "modality": "audio",
                        "subset": "noise",
                        "severity": str(severity),
                        "corruption": corr,
                        "source_path": str(src),
                        "output_path": str(dst),
                        "status": status,
                        "error": error,
                    },
                )
                if sample_offset % 1000 == 0:
                    print(
                        f"[INFO] audio noisy progress total={sample_offset}/{sum(severity_counts.values())}",
                        flush=True,
                    )


def _is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path):
            return True
    except Exception:
        return False


def _build_image_set(
    source_dir: Path,
    out_root: Path,
    rng: random.Random,
    clean_count: int,
    severity_counts: dict[int, int],
    clean_mode: str,
    overwrite: bool,
):
    image_files = [path for path in _iter_files(source_dir, IMAGE_EXTENSIONS) if _is_valid_image(path)]
    if not image_files:
        raise RuntimeError(f"No valid source images found in {source_dir}")

    modality_root = out_root / "image_1m"
    clean_root = modality_root / "clean"
    metadata_path = modality_root / "metadata.csv"
    _write_metadata_header(metadata_path)

    print(f"[INFO] image source files={len(image_files)}", flush=True)

    for idx in range(clean_count):
        src = rng.choice(image_files)
        dst = clean_root / f"sample_{idx:07d}{src.suffix.lower()}"
        try:
            _materialize_clean_file(src, dst, mode=clean_mode, overwrite=overwrite)
            status = "ok"
            error = ""
        except Exception as exc:
            status = "error"
            error = str(exc)
        _append_metadata_row(
            metadata_path,
            {
                "sample_id": f"clean_{idx:07d}",
                "modality": "image",
                "subset": "clean",
                "severity": "0",
                "corruption": "",
                "source_path": str(src),
                "output_path": str(dst),
                "status": status,
                "error": error,
            },
        )
        if idx % 5000 == 0:
            print(f"[INFO] image clean progress {idx}/{clean_count}", flush=True)

    sample_offset = 0
    for severity in (1, 2, 3, 4, 5):
        target = severity_counts[severity]
        per_corr = _allocate_across_corruptions(target, IMAGE_CORRUPTIONS)
        for corr in IMAGE_CORRUPTIONS:
            corr_root = modality_root / f"I={corr}_S={severity}"
            corr_target = per_corr[corr]
            for _ in range(corr_target):
                src = rng.choice(image_files)
                dst = corr_root / f"sample_{sample_offset:07d}{src.suffix.lower()}"
                sample_offset += 1
                if dst.exists() and not overwrite:
                    status = "skipped_exists"
                    error = ""
                else:
                    try:
                        apply_image_corruption(src, dst, corr, severity)
                        status = "ok"
                        error = ""
                    except Exception as exc:
                        status = "error"
                        error = str(exc)
                _append_metadata_row(
                    metadata_path,
                    {
                        "sample_id": f"noise_{sample_offset:07d}",
                        "modality": "image",
                        "subset": "noise",
                        "severity": str(severity),
                        "corruption": corr,
                        "source_path": str(src),
                        "output_path": str(dst),
                        "status": status,
                        "error": error,
                    },
                )
                if sample_offset % 5000 == 0:
                    print(
                        f"[INFO] image noisy progress total={sample_offset}/{sum(severity_counts.values())}",
                        flush=True,
                    )


def _load_text_chunk_pool(
    source_dir: Path,
    extensions: set[str],
    min_chars: int,
    max_chars: int,
    max_chunks: int,
) -> list[str]:
    pool: list[str] = []
    text_files = list(_iter_files(source_dir, extensions))
    if not text_files:
        raise RuntimeError(f"No source text files found in {source_dir}")

    for path in text_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for chunk in _iter_text_chunks(text, min_chars=min_chars, max_chars=max_chars):
            pool.append(chunk)
            if len(pool) >= max_chunks:
                return pool
    return pool


def _build_text_set(
    source_dir: Path,
    out_root: Path,
    rng: random.Random,
    clean_count: int,
    severity_counts: dict[int, int],
    min_chars: int,
    max_chars: int,
    max_pool_chunks: int,
    extensions: set[str],
    overwrite: bool,
):
    chunk_pool = _load_text_chunk_pool(
        source_dir,
        extensions=extensions,
        min_chars=min_chars,
        max_chars=max_chars,
        max_chunks=max_pool_chunks,
    )
    if not chunk_pool:
        raise RuntimeError("No text chunks available after parsing source files.")

    modality_root = out_root / "text_1m"
    clean_root = modality_root / "clean"
    metadata_path = modality_root / "metadata.csv"
    _write_metadata_header(metadata_path)

    print(f"[INFO] text source chunks={len(chunk_pool)}", flush=True)

    for idx in range(clean_count):
        chunk = rng.choice(chunk_pool)
        dst = clean_root / f"sample_{idx:07d}.txt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not overwrite:
            status = "skipped_exists"
            error = ""
        else:
            try:
                dst.write_text(chunk, encoding="utf-8")
                status = "ok"
                error = ""
            except Exception as exc:
                status = "error"
                error = str(exc)

        _append_metadata_row(
            metadata_path,
            {
                "sample_id": f"clean_{idx:07d}",
                "modality": "text",
                "subset": "clean",
                "severity": "0",
                "corruption": "",
                "source_path": "<text_chunk_pool>",
                "output_path": str(dst),
                "status": status,
                "error": error,
            },
        )
        if idx % 10000 == 0:
            print(f"[INFO] text clean progress {idx}/{clean_count}", flush=True)

    sample_offset = 0
    for severity in (1, 2, 3, 4, 5):
        target = severity_counts[severity]
        per_corr = _allocate_across_corruptions(target, TEXT_CORRUPTIONS)
        for corr in TEXT_CORRUPTIONS:
            corr_root = modality_root / f"T={corr}_S={severity}"
            corr_target = per_corr[corr]
            for _ in range(corr_target):
                src_chunk = rng.choice(chunk_pool)
                dst = corr_root / f"sample_{sample_offset:07d}.txt"
                sample_offset += 1
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists() and not overwrite:
                    status = "skipped_exists"
                    error = ""
                else:
                    try:
                        degraded = perturb(src_chunk, corr, severity, rng)
                        dst.write_text(degraded, encoding="utf-8")
                        status = "ok"
                        error = ""
                    except Exception as exc:
                        status = "error"
                        error = str(exc)
                _append_metadata_row(
                    metadata_path,
                    {
                        "sample_id": f"noise_{sample_offset:07d}",
                        "modality": "text",
                        "subset": "noise",
                        "severity": str(severity),
                        "corruption": corr,
                        "source_path": "<text_chunk_pool>",
                        "output_path": str(dst),
                        "status": status,
                        "error": error,
                    },
                )
                if sample_offset % 10000 == 0:
                    print(
                        f"[INFO] text noisy progress total={sample_offset}/{sum(severity_counts.values())}",
                        flush=True,
                    )


def main():
    args = parse_args()
    modalities = _parse_modalities(args.modalities)
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    severity_counts = {
        1: args.severity_1_count,
        2: args.severity_2_count,
        3: args.severity_3_count,
        4: args.severity_4_count,
        5: args.severity_5_count,
    }
    for severity, count in severity_counts.items():
        if count < 0:
            raise ValueError(f"Severity count cannot be negative: s{severity}={count}")
    if args.clean_count < 0:
        raise ValueError("clean_count cannot be negative.")

    total_target = args.clean_count + sum(severity_counts.values())
    print(f"[INFO] target_per_modality={total_target}", flush=True)
    print(
        "[INFO] target_breakdown "
        f"clean={args.clean_count} s1={severity_counts[1]} s2={severity_counts[2]} "
        f"s3={severity_counts[3]} s4={severity_counts[4]} s5={severity_counts[5]}",
        flush=True,
    )

    if shutil.which("ffmpeg") is None and "audio" in modalities:
        raise RuntimeError("ffmpeg is required for audio corruption generation but not found on PATH.")

    text_extensions = _parse_extensions(args.text_extensions)

    if "audio" in modalities:
        if not args.audio_source_dir:
            raise ValueError("--audio-source-dir is required when modalities include audio.")
        rng_audio = random.Random(args.seed + 11)
        _build_audio_set(
            source_dir=Path(args.audio_source_dir).expanduser().resolve(),
            out_root=out_root,
            rng=rng_audio,
            clean_count=args.clean_count,
            severity_counts=severity_counts,
            clean_mode=args.clean_link_mode,
            overwrite=args.overwrite,
            audio_sr=args.audio_sr,
        )
        print("[INFO] finished modality=audio", flush=True)

    if "image" in modalities:
        if not args.image_source_dir:
            raise ValueError("--image-source-dir is required when modalities include image.")
        rng_image = random.Random(args.seed + 23)
        _build_image_set(
            source_dir=Path(args.image_source_dir).expanduser().resolve(),
            out_root=out_root,
            rng=rng_image,
            clean_count=args.clean_count,
            severity_counts=severity_counts,
            clean_mode=args.clean_link_mode,
            overwrite=args.overwrite,
        )
        print("[INFO] finished modality=image", flush=True)

    if "text" in modalities:
        if not args.text_source_dir:
            raise ValueError("--text-source-dir is required when modalities include text.")
        rng_text = random.Random(args.seed + 37)
        _build_text_set(
            source_dir=Path(args.text_source_dir).expanduser().resolve(),
            out_root=out_root,
            rng=rng_text,
            clean_count=args.clean_count,
            severity_counts=severity_counts,
            min_chars=args.text_min_chars,
            max_chars=args.text_max_chars,
            max_pool_chunks=args.text_pool_max_chunks,
            extensions=text_extensions,
            overwrite=args.overwrite,
        )
        print("[INFO] finished modality=text", flush=True)

    print(f"[INFO] done. generated_root={out_root}", flush=True)


if __name__ == "__main__":
    main()
