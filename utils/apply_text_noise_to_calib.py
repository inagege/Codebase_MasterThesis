#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

from apply_all_text_noise import TEXT_CORRUPTIONS, perturb


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Apply text perturbations to plain-text files and write degraded copies under "
            "T=<corruption>_S=<severity> folders."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing text files.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory root for degraded variants.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="chunked",
        choices=["chunked", "full_per_severity"],
        help=(
            "chunked: split each file into 6 parts (clean + severity 1..5) per corruption; "
            "full_per_severity: generate full-file outputs for each severity."
        ),
    )
    parser.add_argument(
        "--severities",
        type=str,
        default="3",
        help="Comma-separated severity values (1-5), for example: 1,2,3,4,5",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Base random seed.",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=".txt,.raw,.md,.tokens",
        help="Comma-separated file suffixes to perturb.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing degraded files.",
    )
    return parser.parse_args()


def _parse_severities(raw: str) -> list[int]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        severity = int(token)
        if severity < 1 or severity > 5:
            raise ValueError(f"Severity must be in [1, 5], got {severity}")
        values.append(severity)
    if not values:
        raise ValueError("No valid severities provided.")
    # Keep user-provided order but drop duplicates.
    dedup = []
    seen = set()
    for severity in values:
        if severity in seen:
            continue
        seen.add(severity)
        dedup.append(severity)
    return dedup


def _parse_extensions(raw: str) -> set[str]:
    exts = set()
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if not token.startswith("."):
            token = f".{token}"
        exts.add(token)
    if not exts:
        raise ValueError("No file extensions provided.")
    return exts


def _iter_text_files(input_dir: Path, extensions: set[str]):
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("._"):
            continue
        if path.suffix.lower() in extensions:
            yield path


def _apply_chunked_mixture(text: str, corruption: str, rng: random.Random) -> str:
    if not text:
        return text
    n = len(text)
    boundaries = [0]
    for i in range(1, 6):
        boundaries.append((n * i) // 6)
    boundaries.append(n)

    out_parts: list[str] = []
    for idx in range(6):
        start = boundaries[idx]
        end = boundaries[idx + 1]
        segment = text[start:end]
        if idx == 0:
            out_parts.append(segment)
        else:
            out_parts.append(perturb(segment, corruption, idx, rng))
    return "".join(out_parts)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    severities = _parse_severities(args.severities)
    extensions = _parse_extensions(args.extensions)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    text_files = list(_iter_text_files(input_dir, extensions))
    if not text_files:
        raise RuntimeError(
            f"No matching text files found in {input_dir} for extensions {sorted(extensions)}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    total_written = 0

    if args.mode == "chunked":
        for corruption_idx, corruption in enumerate(TEXT_CORRUPTIONS):
            combo_root = out_dir / f"T={corruption}_S=mixed"
            combo_root.mkdir(parents=True, exist_ok=True)

            combo_seed = args.seed + corruption_idx * 100000
            rng = random.Random(combo_seed)
            written_this_combo = 0

            for file_idx, src in enumerate(text_files, start=1):
                rel = src.relative_to(input_dir)
                dst = combo_root / rel
                if dst.exists() and not args.overwrite:
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                print(
                    f"[INFO] variant={combo_root.name} processing={file_idx}/{len(text_files)} file={rel}",
                    flush=True,
                )

                text = src.read_text(encoding="utf-8", errors="ignore")
                degraded = _apply_chunked_mixture(text, corruption, rng)
                dst.write_text(degraded, encoding="utf-8")
                written_this_combo += 1
                total_written += 1

            print(
                f"[INFO] variant={combo_root.name} files={len(text_files)} written={written_this_combo}",
                flush=True,
            )
    else:
        for severity in severities:
            for corruption_idx, corruption in enumerate(TEXT_CORRUPTIONS):
                combo_root = out_dir / f"T={corruption}_S={severity}"
                combo_root.mkdir(parents=True, exist_ok=True)

                combo_seed = args.seed + severity * 1000 + corruption_idx * 100000
                rng = random.Random(combo_seed)
                written_this_combo = 0

                for file_idx, src in enumerate(text_files, start=1):
                    rel = src.relative_to(input_dir)
                    dst = combo_root / rel
                    if dst.exists() and not args.overwrite:
                        continue
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    print(
                        f"[INFO] variant={combo_root.name} processing={file_idx}/{len(text_files)} file={rel}",
                        flush=True,
                    )

                    text = src.read_text(encoding="utf-8", errors="ignore")
                    degraded = perturb(text, corruption, severity, rng)
                    dst.write_text(degraded, encoding="utf-8")
                    written_this_combo += 1
                    total_written += 1

                print(
                    f"[INFO] variant={combo_root.name} files={len(text_files)} written={written_this_combo}",
                    flush=True,
                )

    print(f"[INFO] Done. total_written={total_written} out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()
