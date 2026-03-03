#!/usr/bin/env python3
import argparse
import csv
import subprocess
import sys
from pathlib import Path


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


def _run(cmd: list[str], error_csv: Path | None = None):
    print("[RUN]", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:
        print(f"[WARN] Failed command: {' '.join(cmd)}: {exc}", flush=True)
        if error_csv is not None:
            _append_error_row(error_csv, cmd[-1] if cmd else "<unknown>", str(exc))
        return False
    return True


def _parse_modalities(modalities: str | None, dataset: str) -> set[str]:
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


def _run_meld(args, modalities: set[str]):
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


def _run_homeprice(args, modalities: set[str]):
    error_csv = Path(args.error_csv)
    csv_candidates = [
        Path("data/HomePrice/data_price_binned.csv"),
        Path("data/HomePrice/data_prive_binned.csv"),
    ]
    csv_path = next((p for p in csv_candidates if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError("HomePrice CSV not found.")
    out_root = Path("data/HomePrice/noise")

    if "image" in modalities:
        cmd = [
            sys.executable,
            "utils/apply_all_image_noise.py",
            "--images_dir",
            "data/HomePrice/homeImages",
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
            str(csv_path),
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


def _run_imdb(args, modalities: set[str]):
    error_csv = Path(args.error_csv)
    out_root = Path("data/IMDB/noise")
    if "image" in modalities:
        cmd = [
            sys.executable,
            "utils/apply_all_image_noise.py",
            "--images_dir",
            "data/IMDB/IMDB_four_genre_posters",
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
            "data/IMDB/IMDB_four_genre_larger_plot_description.csv",
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


def _run_voxceleb(args, modalities: set[str]):
    error_csv = Path(args.error_csv)
    videos_dir = Path("data/VoxCeleb2/dev/mp4")
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


def _run_nejm(args, modalities: set[str]):
    error_csv = Path(args.error_csv)
    out_root = Path("data/NEJM/noise")
    if "image" in modalities:
        cmd = [
            sys.executable,
            "utils/apply_all_image_noise.py",
            "--images_dir",
            "data/NEJM/images",
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
            "data/NEJM/metadata.csv",
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


def _run_marine(args, modalities: set[str]):
    error_csv = Path(args.error_csv)
    out_root = Path("data/Marine/noise")
    if "image" in modalities:
        cmd = [
            sys.executable,
            "utils/apply_all_image_noise.py",
            "--images_dir",
            "data/Marine/images",
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
            "data/Marine/audio",
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
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--error-csv", default="error.csv", help="Path to append errors as file,error.")
    args = ap.parse_args()

    modalities = _parse_modalities(args.modalities, args.dataset)

    if args.dataset == "meld":
        _run_meld(args, modalities)
    elif args.dataset == "homeprice":
        _run_homeprice(args, modalities)
    elif args.dataset == "imdb":
        _run_imdb(args, modalities)
    elif args.dataset == "voxceleb":
        _run_voxceleb(args, modalities)
    elif args.dataset == "nejm":
        _run_nejm(args, modalities)
    elif args.dataset == "marine":
        _run_marine(args, modalities)
    else:
        raise ValueError(f"Unsupported dataset {args.dataset}")

    print("[DONE] Dataset noise generation finished.")


if __name__ == "__main__":
    main()
