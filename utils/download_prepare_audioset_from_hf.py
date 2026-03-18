from __future__ import annotations

import argparse
import csv
import shutil
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus", ".webm"}
SUPPORTED_SUBSETS = {"20k", "500k", "2m"}
SUPPORTED_SPLITS = {"train", "test"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download AudioSet WebDataset shards from Hugging Face and extract audio files "
            "into an import-ready folder structure."
        )
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="confit/audioset-16khz-wds",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="500k",
        choices=sorted(SUPPORTED_SUBSETS),
        help="Dataset subset to download.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,test",
        help="Comma-separated split list from train,test.",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="data/calibration_data/sources/audioset",
        help="Output root containing downloaded shards and extracted clips.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default="",
        help="Optional HF token for private/rate-limited access.",
    )
    parser.add_argument(
        "--extract-workers",
        type=int,
        default=8,
        help="Parallel workers for shard extraction.",
    )
    parser.add_argument(
        "--max-shards-per-split",
        type=int,
        default=0,
        help="Optional cap for debugging. Use 0 to extract all shards.",
    )
    parser.add_argument(
        "--overwrite-extracted",
        action="store_true",
        help="Overwrite existing audio files if names collide.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip HF download and only run extraction from local shard folders.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction and only download shards.",
    )
    return parser.parse_args()


def _parse_splits(splits_arg: str) -> list[str]:
    splits = [token.strip().lower() for token in splits_arg.split(",") if token.strip()]
    if not splits:
        raise ValueError("No split selected. Use train,test.")
    invalid = sorted(set(splits) - SUPPORTED_SPLITS)
    if invalid:
        raise ValueError(f"Unsupported splits: {invalid}. Supported: {sorted(SUPPORTED_SPLITS)}")
    deduped: list[str] = []
    for split in splits:
        if split not in deduped:
            deduped.append(split)
    return deduped


def _download_shards(
    repo_id: str,
    subset: str,
    splits: list[str],
    wds_root: Path,
    hf_token: str | None,
):
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "huggingface_hub is required. Install it (e.g., via pixi/conda/pip) "
            "or run in an environment that includes transformers/huggingface_hub."
        ) from exc

    allow_patterns = [f"{subset}/{split}/*.tar" for split in splits]
    print(
        f"[INFO] Downloading HF dataset repo={repo_id} subset={subset} patterns={allow_patterns}",
        flush=True,
    )
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(wds_root),
        allow_patterns=allow_patterns,
        token=hf_token,
        resume_download=True,
    )


def _resolve_destination_path(
    split_out_dir: Path,
    shard_stem: str,
    member_name: str,
    overwrite: bool,
) -> tuple[Path, bool]:
    base_name = Path(member_name).name
    if not base_name:
        return split_out_dir / "", True

    destination = split_out_dir / base_name
    if not destination.exists():
        return destination, False

    if overwrite:
        return destination, False

    prefixed = split_out_dir / f"{shard_stem}__{base_name}"
    if not prefixed.exists():
        return prefixed, False

    index = 1
    while True:
        candidate = split_out_dir / f"{shard_stem}__{index}__{base_name}"
        if not candidate.exists():
            return candidate, False
        index += 1


def _marker_path(markers_root: Path, subset: str, split: str, shard_path: Path) -> Path:
    return markers_root / subset / split / f"{shard_path.stem}.done"


def _extract_one_shard(
    shard_path: Path,
    split: str,
    subset: str,
    clips_root: Path,
    markers_root: Path,
    overwrite_extracted: bool,
) -> dict[str, int | str]:
    marker = _marker_path(markers_root, subset, split, shard_path)
    if marker.exists() and not overwrite_extracted:
        return {
            "split": split,
            "shard": str(shard_path),
            "status": "skipped_done",
            "audio_files_extracted": 0,
            "non_audio_members": 0,
            "errors": 0,
        }

    split_out_dir = clips_root / subset / split
    split_out_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    non_audio = 0
    errors = 0

    try:
        with tarfile.open(shard_path, "r") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                suffix = Path(member.name).suffix.lower()
                if suffix not in AUDIO_EXTENSIONS:
                    non_audio += 1
                    continue

                destination, invalid = _resolve_destination_path(
                    split_out_dir=split_out_dir,
                    shard_stem=shard_path.stem,
                    member_name=member.name,
                    overwrite=overwrite_extracted,
                )
                if invalid:
                    errors += 1
                    continue

                fileobj = archive.extractfile(member)
                if fileobj is None:
                    errors += 1
                    continue

                if overwrite_extracted and destination.exists():
                    destination.unlink()
                with destination.open("wb") as out_handle:
                    shutil.copyfileobj(fileobj, out_handle)
                extracted += 1
    except Exception:
        errors += 1
        return {
            "split": split,
            "shard": str(shard_path),
            "status": "failed",
            "audio_files_extracted": extracted,
            "non_audio_members": non_audio,
            "errors": errors,
        }

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"audio_files_extracted={extracted}\n", encoding="utf-8")
    return {
        "split": split,
        "shard": str(shard_path),
        "status": "ok",
        "audio_files_extracted": extracted,
        "non_audio_members": non_audio,
        "errors": errors,
    }


def _write_report(report_rows: list[dict[str, int | str]], report_path: Path):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "shard",
        "status",
        "audio_files_extracted",
        "non_audio_members",
        "errors",
    ]
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report_rows:
            writer.writerow(row)


def main():
    args = parse_args()
    splits = _parse_splits(args.splits)

    out_root = Path(args.out_root).expanduser().resolve()
    wds_root = out_root / "wds"
    clips_root = out_root / "clips"
    markers_root = out_root / "metadata" / "extracted_shards"
    report_path = out_root / "metadata" / "extraction_report.csv"
    wds_root.mkdir(parents=True, exist_ok=True)
    clips_root.mkdir(parents=True, exist_ok=True)

    hf_token = args.hf_token.strip() or None
    if not args.skip_download:
        _download_shards(
            repo_id=args.repo_id,
            subset=args.subset,
            splits=splits,
            wds_root=wds_root,
            hf_token=hf_token,
        )

    if args.skip_extract:
        print("[INFO] Skipping extraction as requested.", flush=True)
        print(f"[INFO] Shard root: {wds_root / args.subset}", flush=True)
        return

    tasks: list[tuple[str, Path]] = []
    for split in splits:
        split_dir = wds_root / args.subset / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Expected split directory not found: {split_dir}. "
                "If download was skipped, ensure shards are present."
            )
        shard_paths = sorted(split_dir.glob("*.tar"))
        if args.max_shards_per_split > 0:
            shard_paths = shard_paths[: args.max_shards_per_split]
        if not shard_paths:
            raise RuntimeError(f"No shard tar files found in {split_dir}")
        print(
            f"[INFO] Extraction input split={split} subset={args.subset} shards={len(shard_paths)}",
            flush=True,
        )
        tasks.extend((split, shard_path) for shard_path in shard_paths)

    report_rows: list[dict[str, int | str]] = []
    workers = max(1, args.extract_workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                _extract_one_shard,
                shard_path,
                split,
                args.subset,
                clips_root,
                markers_root,
                args.overwrite_extracted,
            ): (split, shard_path)
            for split, shard_path in tasks
        }

        completed = 0
        total = len(future_map)
        progress_every = max(5, total // 20)
        for future in as_completed(future_map):
            row = future.result()
            report_rows.append(row)
            completed += 1
            if completed == total or completed % progress_every == 0:
                print(f"[INFO] Extraction progress {completed}/{total}", flush=True)

    report_rows.sort(key=lambda row: (str(row["split"]), str(row["shard"])))
    _write_report(report_rows, report_path)

    status_counts: dict[str, int] = {}
    extracted_total = 0
    errors_total = 0
    for row in report_rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        extracted_total += int(row["audio_files_extracted"])
        errors_total += int(row["errors"])

    counts_msg = ", ".join(f"{status}={count}" for status, count in sorted(status_counts.items()))
    print(f"[INFO] Extraction report: {report_path}", flush=True)
    print(f"[INFO] Shard status counts: {counts_msg}", flush=True)
    print(f"[INFO] Extracted audio files (this run): {extracted_total}", flush=True)
    print(f"[INFO] Extraction errors: {errors_total}", flush=True)
    print(f"[INFO] Prepared clip root for import: {clips_root / args.subset}", flush=True)


if __name__ == "__main__":
    main()
