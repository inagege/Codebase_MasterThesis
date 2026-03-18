#!/usr/bin/env python3
import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MELD_SPLIT_ROOT = {
    "train": Path("data/MELD.Raw/train_splits"),
    "test": Path("data/MELD.Raw/output_repeated_splits_test"),
    "val": Path("data/MELD.Raw/dev_splits_complete"),
}
MELD_SPLIT_META = {
    "train": "train_sent_emo.csv",
    "test": "test_sent_emo.csv",
    "val": "dev_sent_emo.csv",
}

DATASET_MODALITIES = {
    "meld": {"text", "audio", "video"},
    "homeprice": {"text", "image"},
    "imdb": {"text", "image"},
    "voxceleb": {"audio", "video"},
    "nejm": {"text", "image"},
    "marine": {"audio", "image"},
}


def _append_error_row(path: Path, file_name: str, error: str):
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "error"])
        if write_header:
            writer.writeheader()
        writer.writerow({"file": file_name, "error": error})


def _run(cmd: list[str], error_csv: Optional[Path] = None):
    print("[RUN]", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:
        print(f"[WARN] Failed command: {' '.join(cmd)}: {exc}", flush=True)
        if error_csv is not None:
            _append_error_row(error_csv, cmd[-1] if cmd else "<unknown>", str(exc))
        return False
    return True


def _parse_modalities(modalities: Optional[str], dataset: str) -> set[str]:
    allowed = DATASET_MODALITIES[dataset]
    if modalities is None:
        return set(allowed)
    mods = {m.strip().lower() for m in modalities.split(",") if m.strip()}
    bad = mods - {"text", "audio", "video", "image"}
    if bad:
        raise ValueError(f"Unknown modalities: {sorted(bad)}")
    unsupported = mods - allowed
    if unsupported:
        raise ValueError(
            f"Modalities {sorted(unsupported)} are not available for dataset {dataset}. "
            f"Allowed: {sorted(allowed)}"
        )
    if not mods:
        raise ValueError("No modalities selected.")
    return mods


def _normalize_meld_splits(split_arg: str) -> list[str]:
    requested = {s.strip().lower() for s in split_arg.split(",") if s.strip()}
    if not requested:
        requested = {"test"}

    result = set()
    for split in requested:
        if split in {"all", "*"}:
            result.update({"train", "val", "test"})
            continue
        if split == "dev":
            split = "val"
        if split not in {"train", "val", "test"}:
            raise ValueError("For MELD, use split from: train,val,test,dev,all.")
        result.add(split)
    return sorted(result)


def _add_common_flags(cmd: list[str], recursive: bool, overwrite: bool):
    if recursive:
        cmd.append("--recursive")
    if overwrite:
        cmd.append("--overwrite")


def _sanitize_token(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _collect_paths_by_modality(samples: list[dict]) -> dict[str, set[Path]]:
    paths = {"image": set(), "video": set(), "audio": set()}
    for sample in samples:
        for modality, key in (("image", "image"), ("video", "video"), ("audio", "audio")):
            path_value = sample.get(key)
            if path_value:
                paths[modality].add(Path(path_value))
    return paths


def _build_stratified_selection(args, modalities: set[str]):
    if args.stratified_samples is None:
        return None

    if args.dataset == "meld":
        print(
            "[INFO] Ignoring --stratified-samples for MELD (explicit train/val/test splits are already defined).",
            flush=True,
        )
        return None

    try:
        from utils.benchmark_data_loading import load_samples, select_stratified_samples
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Stratified sampling requires benchmark data-loading dependencies (including pandas). "
            "Please run this script in the project environment (e.g. via `pixi run ...`)."
        ) from exc

    label_column = "nationality_wiki" if args.dataset == "voxceleb" else None
    base_samples = load_samples(args.dataset, args, modalities, None, label_column)
    total = len(base_samples)
    if total == 0:
        raise RuntimeError("No base samples available for stratified sampling.")

    if args.stratified_samples >= total:
        selected = base_samples
        print(
            "[INFO] --stratified-samples is >= available samples; selecting all samples.",
            flush=True,
        )
    else:
        selected = select_stratified_samples(base_samples, args.stratified_samples)
        print(
            f"[INFO] Applied benchmark-equivalent deterministic stratified sampling: {total} -> {len(selected)} samples",
            flush=True,
        )

    selected_ids = {
        sample_id
        for sample_id in (_sanitize_token(sample.get("sample_id")) for sample in selected)
        if sample_id
    }
    paths_by_modality = _collect_paths_by_modality(selected)
    print(
        "[INFO] Stratified subset summary: "
        f"sample_ids={len(selected_ids)} image_files={len(paths_by_modality['image'])} "
        f"video_files={len(paths_by_modality['video'])} audio_files={len(paths_by_modality['audio'])}",
        flush=True,
    )
    print(
        "[INFO] Sample selection uses the same deterministic selector as benchmark_scored_modalities.py "
        "(seed-independent).",
        flush=True,
    )
    return {
        "sample_ids": selected_ids,
        "paths_by_modality": paths_by_modality,
    }


def _relative_to_or_none(path: Path, root: Path):
    try:
        return path.resolve().relative_to(root.resolve())
    except Exception:
        return None


def _link_or_copy(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except Exception:
        shutil.copy2(src, dst)


def _stage_selected_files(
    source_root: Path,
    selected_paths: set[Path],
    staging_root: Path,
    stage_name: str,
) -> Path:
    if not selected_paths:
        raise RuntimeError(f"No selected files available for stage '{stage_name}'.")

    staged_dir = staging_root / stage_name
    staged_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src_path in sorted((Path(p) for p in selected_paths), key=lambda p: p.as_posix()):
        if not src_path.exists():
            print(f"[WARN] Skipping missing selected file: {src_path}", flush=True)
            continue
        rel = _relative_to_or_none(src_path, source_root)
        if rel is None:
            continue
        dst = staged_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        _link_or_copy(src_path, dst)
        copied += 1

    if copied == 0:
        raise RuntimeError(
            f"No selected files from '{source_root}' could be staged into '{staged_dir}'."
        )
    print(
        f"[INFO] Staged {copied} selected files from {source_root} into {staged_dir}",
        flush=True,
    )
    return staged_dir


def _selected_paths_for_modality(
    selection,
    modality: str,
    source_root: Path,
) -> set[Path]:
    if selection is None:
        return set()

    paths = selection["paths_by_modality"].get(modality, set())
    result = set()
    for path in paths:
        rel = _relative_to_or_none(Path(path), source_root)
        if rel is not None:
            result.add(source_root / rel)
    return result


def _stage_filtered_csv(
    input_csv: Path,
    key_column: str,
    selected_ids: set[str],
    staging_root: Path,
    stage_name: str,
) -> Path:
    if not selected_ids:
        raise RuntimeError("No selected sample IDs available for CSV filtering.")

    out_csv = staging_root / f"{stage_name}_{input_csv.name}"
    total_rows = 0
    kept_rows = 0
    with open(input_csv, "r", newline="", encoding="utf-8") as in_handle:
        reader = csv.DictReader(in_handle)
        if key_column not in (reader.fieldnames or []):
            raise ValueError(
                f"CSV {input_csv} does not contain key column {key_column!r}. "
                f"Found: {reader.fieldnames}"
            )

        with open(out_csv, "w", newline="", encoding="utf-8") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                total_rows += 1
                if _sanitize_token(row.get(key_column)) in selected_ids:
                    writer.writerow(row)
                    kept_rows += 1

    print(
        f"[INFO] Filtered CSV {input_csv} on {key_column}: {total_rows} -> {kept_rows} rows",
        flush=True,
    )
    if kept_rows == 0:
        raise RuntimeError(f"Filtered CSV for {input_csv} is empty.")
    return out_csv


def _selected_voxceleb_video_paths(selection, videos_dir: Path) -> set[Path]:
    if selection is None:
        return set()

    direct_video_paths = _selected_paths_for_modality(selection, "video", videos_dir)
    if direct_video_paths:
        return direct_video_paths

    derived = set()
    for sample_id in selection["sample_ids"]:
        candidate = videos_dir / f"{sample_id}.mp4"
        if candidate.exists():
            derived.add(candidate)
    return derived


def _run_meld(args, modalities: set[str], selection, staging_root: Path):
    _ = selection
    _ = staging_root
    error_csv = Path(args.error_csv)
    splits = _normalize_meld_splits(args.split)
    for split in splits:
        root = MELD_SPLIT_ROOT[split]
        videos_dir = root / "unmodified"
        text_csv = root / MELD_SPLIT_META[split]

        if "video" in modalities:
            cmd = [
                sys.executable,
                "utils/apply_all_visual_noise.py",
                "--videos_dir",
                str(videos_dir),
                "--out_dir",
                str(root / "video"),
                "--severity",
                str(args.severity),
            ]
            _add_common_flags(cmd, recursive=args.recursive, overwrite=args.overwrite)
            _run(cmd, error_csv=error_csv)

        if "audio" in modalities:
            cmd = [
                sys.executable,
                "utils/apply_all_audio_noise.py",
                "--videos_dir",
                str(videos_dir),
                "--out_dir",
                str(root / "audio"),
                "--severity",
                str(args.severity),
            ]
            _add_common_flags(cmd, recursive=args.recursive, overwrite=args.overwrite)
            _run(cmd, error_csv=error_csv)

        if "text" in modalities:
            cmd = [
                sys.executable,
                "utils/apply_all_text_noise.py",
                "--input_csv",
                str(text_csv),
                "--out_dir",
                str(root / "text"),
                "--severity",
                str(args.severity),
                "--seed",
                str(args.seed),
                "--text-column",
                "Utterance",
                "--output-filename",
                "metadata.csv",
            ]
            _run(cmd, error_csv=error_csv)


def _run_homeprice(args, modalities: set[str], selection, staging_root: Path):
    error_csv = Path(args.error_csv)
    csv_candidates = [
        Path("data/HomePrice/data_price_binned.csv"),
        Path("data/HomePrice/data_prive_binned.csv"),
    ]
    csv_path = next((p for p in csv_candidates if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError("HomePrice CSV not found.")

    images_dir = Path("data/HomePrice/homeImages")
    text_csv = csv_path
    if selection is not None and "image" in modalities:
        selected_images = _selected_paths_for_modality(selection, "image", images_dir)
        images_dir = _stage_selected_files(
            images_dir,
            selected_images,
            staging_root,
            "homeprice_images",
        )
    if selection is not None and "text" in modalities:
        text_csv = _stage_filtered_csv(
            input_csv=csv_path,
            key_column="homeImage",
            selected_ids=selection["sample_ids"],
            staging_root=staging_root,
            stage_name="homeprice_text",
        )

    out_root = Path("data/HomePrice/noise")
    if "image" in modalities:
        cmd = [
            sys.executable,
            "utils/apply_all_image_noise.py",
            "--images_dir",
            str(images_dir),
            "--out_dir",
            str(out_root / "image"),
            "--severity",
            str(args.severity),
        ]
        _add_common_flags(cmd, recursive=args.recursive, overwrite=args.overwrite)
        _run(cmd, error_csv=error_csv)

    if "text" in modalities:
        cmd = [
            sys.executable,
            "utils/apply_all_text_noise.py",
            "--input_csv",
            str(text_csv),
            "--out_dir",
            str(out_root / "text"),
            "--severity",
            str(args.severity),
            "--seed",
            str(args.seed),
            "--text-column",
            "description",
            "--output-filename",
            csv_path.name,
        ]
        _run(cmd, error_csv=error_csv)


def _run_imdb(args, modalities: set[str], selection, staging_root: Path):
    error_csv = Path(args.error_csv)
    images_dir = Path("data/IMDB/IMDB_four_genre_posters")
    text_csv = Path("data/IMDB/IMDB_four_genre_larger_plot_description.csv")

    if selection is not None and "image" in modalities:
        selected_images = _selected_paths_for_modality(selection, "image", images_dir)
        images_dir = _stage_selected_files(
            images_dir,
            selected_images,
            staging_root,
            "imdb_images",
        )
    if selection is not None and "text" in modalities:
        text_csv = _stage_filtered_csv(
            input_csv=text_csv,
            key_column="movie_id",
            selected_ids=selection["sample_ids"],
            staging_root=staging_root,
            stage_name="imdb_text",
        )

    out_root = Path("data/IMDB/noise")
    if "image" in modalities:
        cmd = [
            sys.executable,
            "utils/apply_all_image_noise.py",
            "--images_dir",
            str(images_dir),
            "--out_dir",
            str(out_root / "image"),
            "--severity",
            str(args.severity),
        ]
        _add_common_flags(cmd, recursive=args.recursive, overwrite=args.overwrite)
        _run(cmd, error_csv=error_csv)

    if "text" in modalities:
        cmd = [
            sys.executable,
            "utils/apply_all_text_noise.py",
            "--input_csv",
            str(text_csv),
            "--out_dir",
            str(out_root / "text"),
            "--severity",
            str(args.severity),
            "--seed",
            str(args.seed),
            "--text-column",
            "description",
            "--output-filename",
            "IMDB_four_genre_larger_plot_description.csv",
        ]
        _run(cmd, error_csv=error_csv)


def _run_voxceleb(args, modalities: set[str], selection, staging_root: Path):
    error_csv = Path(args.error_csv)
    videos_dir = Path("data/VoxCeleb2/dev/mp4")

    if selection is not None and ("video" in modalities or "audio" in modalities):
        selected_videos = _selected_voxceleb_video_paths(selection, videos_dir)
        videos_dir = _stage_selected_files(
            videos_dir,
            selected_videos,
            staging_root,
            "voxceleb_videos",
        )

    out_root = Path("data/VoxCeleb2/dev/noise")
    if "video" in modalities:
        cmd = [
            sys.executable,
            "utils/apply_all_visual_noise.py",
            "--videos_dir",
            str(videos_dir),
            "--out_dir",
            str(out_root / "video"),
            "--severity",
            str(args.severity),
            "--recursive",
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        _run(cmd, error_csv=error_csv)

    if "audio" in modalities:
        cmd = [
            sys.executable,
            "utils/apply_all_audio_noise.py",
            "--videos_dir",
            str(videos_dir),
            "--out_dir",
            str(out_root / "audio"),
            "--severity",
            str(args.severity),
            "--recursive",
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        _run(cmd, error_csv=error_csv)


def _run_nejm(args, modalities: set[str], selection, staging_root: Path):
    error_csv = Path(args.error_csv)
    images_dir = Path("data/NEJM/images")
    text_csv = Path("data/NEJM/metadata.csv")

    if selection is not None and "image" in modalities:
        selected_images = _selected_paths_for_modality(selection, "image", images_dir)
        images_dir = _stage_selected_files(
            images_dir,
            selected_images,
            staging_root,
            "nejm_images",
        )
    if selection is not None and "text" in modalities:
        text_csv = _stage_filtered_csv(
            input_csv=text_csv,
            key_column="image_id",
            selected_ids=selection["sample_ids"],
            staging_root=staging_root,
            stage_name="nejm_text",
        )

    out_root = Path("data/NEJM/noise")
    if "image" in modalities:
        cmd = [
            sys.executable,
            "utils/apply_all_image_noise.py",
            "--images_dir",
            str(images_dir),
            "--out_dir",
            str(out_root / "image"),
            "--severity",
            str(args.severity),
        ]
        _add_common_flags(cmd, recursive=args.recursive, overwrite=args.overwrite)
        _run(cmd, error_csv=error_csv)

    if "text" in modalities:
        cmd = [
            sys.executable,
            "utils/apply_all_text_noise.py",
            "--input_csv",
            str(text_csv),
            "--out_dir",
            str(out_root / "text"),
            "--severity",
            str(args.severity),
            "--seed",
            str(args.seed),
            "--text-column",
            "question",
            "--output-filename",
            "metadata.csv",
        ]
        _run(cmd, error_csv=error_csv)


def _run_marine(args, modalities: set[str], selection, staging_root: Path):
    error_csv = Path(args.error_csv)
    images_dir = Path("data/Marine/images")
    audio_dir = Path("data/Marine/audio")

    if selection is not None and "image" in modalities:
        selected_images = _selected_paths_for_modality(selection, "image", images_dir)
        images_dir = _stage_selected_files(
            images_dir,
            selected_images,
            staging_root,
            "marine_images",
        )
    if selection is not None and "audio" in modalities:
        selected_audio = _selected_paths_for_modality(selection, "audio", audio_dir)
        audio_dir = _stage_selected_files(
            audio_dir,
            selected_audio,
            staging_root,
            "marine_audio",
        )

    out_root = Path("data/Marine/noise")
    if "image" in modalities:
        cmd = [
            sys.executable,
            "utils/apply_all_image_noise.py",
            "--images_dir",
            str(images_dir),
            "--out_dir",
            str(out_root / "image"),
            "--severity",
            str(args.severity),
        ]
        _add_common_flags(cmd, recursive=args.recursive, overwrite=args.overwrite)
        _run(cmd, error_csv=error_csv)

    if "audio" in modalities:
        # Marine provides standalone audio, so audio perturbation is applied by
        # wrapping each WAV in a temporary video and extracting the corrupted WAV.
        cmd = [
            sys.executable,
            "utils/apply_audio_noise_from_wavs.py",
            "--audio_dir",
            str(audio_dir),
            "--out_dir",
            str(out_root / "audio"),
            "--severity",
            str(args.severity),
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        _run(cmd, error_csv=error_csv)


def main():
    ap = argparse.ArgumentParser("Apply noise for one dataset with dataset-specific wiring.")
    ap.add_argument(
        "--dataset",
        required=True,
        choices=["meld", "homeprice", "imdb", "voxceleb", "nejm", "marine"],
    )
    ap.add_argument("--split", default="test", help="MELD only: train,val,test,dev,all")
    ap.add_argument("--modalities", default=None, help="Comma-separated from text,audio,video,image")
    ap.add_argument("--severity", type=int, default=3)
    ap.add_argument("--seed", type=int, default=123, help="Seed for text perturbation randomness.")
    ap.add_argument(
        "--stratified-samples",
        type=int,
        default=None,
        help="Non-MELD only: deterministically select this many samples, stratified by label.",
    )
    ap.add_argument(
        "--audio-subdir",
        type=str,
        default="audio_only",
        help="VoxCeleb helper: subdirectory containing WAV files (used for sample selection parity).",
    )
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--error-csv", default="error.csv", help="Path to append errors as file,error.")
    args = ap.parse_args()

    if args.stratified_samples is not None and args.stratified_samples < 1:
        raise ValueError("--stratified-samples must be >= 1 when provided.")

    modalities = _parse_modalities(args.modalities, args.dataset)
    print(f"[INFO] seed={args.seed}", flush=True)
    print(f"[INFO] stratified_samples={args.stratified_samples}", flush=True)
    selection = _build_stratified_selection(args, modalities)

    with tempfile.TemporaryDirectory(prefix="noise_selection_") as tmpdir:
        staging_root = Path(tmpdir)
        if args.dataset == "meld":
            _run_meld(args, modalities, selection, staging_root)
        elif args.dataset == "homeprice":
            _run_homeprice(args, modalities, selection, staging_root)
        elif args.dataset == "imdb":
            _run_imdb(args, modalities, selection, staging_root)
        elif args.dataset == "voxceleb":
            _run_voxceleb(args, modalities, selection, staging_root)
        elif args.dataset == "nejm":
            _run_nejm(args, modalities, selection, staging_root)
        elif args.dataset == "marine":
            _run_marine(args, modalities, selection, staging_root)
        else:
            raise ValueError(f"Unsupported dataset {args.dataset}")

    print("[DONE] Dataset noise generation finished.")


if __name__ == "__main__":
    main()
