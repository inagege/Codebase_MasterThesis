from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path


KNOWN_DATASET_MODALITY = {
    "koniq10k": "image",
    "kadid10k": "image",
    "live_challenge": "image",
    "fsd50k": "audio",
    "tau_urban": "audio",
    "esc50": "audio",
    "urbansound8k": "audio",
    "maestro": "audio",
    "musan": "audio",
    "librispeech": "audio",
    "commonvoice": "audio",
    "odaq": "audio",
    "wikipedia": "text",
    "wikitext": "text",
    "wikitext103": "text",
    "c4": "text",
    "kinetics": "video",
    "ucf101": "video",
    "ucf101_ds": "video",
}

SUPPORTED_MODALITIES = {"text", "audio", "image", "video"}
FILE_EXTENSIONS = {
    "text": {".txt", ".jsonl", ".json", ".csv", ".tsv", ".xml", ".tokens", ".md", ".raw"},
    "audio": {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"},
    "image": {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"},
    "video": {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Import external calibration datasets into data/calibration_data by symlink/copy "
            "and build per-modality file manifests."
        )
    )
    parser.add_argument(
        "--dataset-path",
        action="append",
        required=True,
        help=(
            "Dataset mapping in the form name=/absolute/or/relative/path. "
            "Name is used as the target folder and modality inference key."
        ),
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="data/calibration_data",
        help="Target calibration root directory.",
    )
    parser.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="Import mode for dataset roots.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing imported dataset folders.",
    )
    parser.add_argument(
        "--manifest-max-files-per-dataset",
        type=int,
        default=0,
        help=(
            "Optional max files per dataset per modality in generated manifests. "
            "Use 0 to keep all files."
        ),
    )
    return parser.parse_args()


def _parse_dataset_mapping(dataset_mapping: str) -> tuple[str, Path]:
    if "=" not in dataset_mapping:
        raise ValueError(f"Invalid --dataset-path value: {dataset_mapping!r}. Expected name=/path.")
    name, raw_path = dataset_mapping.split("=", 1)
    dataset_name = name.strip().lower().replace(" ", "_")
    if not dataset_name:
        raise ValueError(f"Dataset name is empty in --dataset-path {dataset_mapping!r}")
    source_path = Path(raw_path.strip()).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset source path does not exist: {source_path}")
    if not source_path.is_dir():
        raise ValueError(f"Dataset source path is not a directory: {source_path}")
    return dataset_name, source_path


def _infer_modality(dataset_name: str, source_path: Path) -> str:
    if dataset_name in KNOWN_DATASET_MODALITY:
        return KNOWN_DATASET_MODALITY[dataset_name]

    # Fallback: infer modality by extension majority.
    extension_counts = {modality: 0 for modality in SUPPORTED_MODALITIES}
    scanned = 0
    for path in source_path.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        for modality, extensions in FILE_EXTENSIONS.items():
            if suffix in extensions:
                extension_counts[modality] += 1
                break
        scanned += 1
        if scanned >= 50000:
            break

    best_modality = max(extension_counts, key=extension_counts.get)
    if extension_counts[best_modality] < 1:
        raise ValueError(
            f"Could not infer modality for dataset {dataset_name!r}; no known file extensions found in {source_path}."
        )
    return best_modality


def _link_or_copy_dataset(source: Path, destination: Path, mode: str):
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    if mode == "symlink":
        os.symlink(source, destination, target_is_directory=True)
        return

    shutil.copytree(source, destination)


def _iter_matching_files(root: Path, modality: str):
    valid_ext = FILE_EXTENSIONS[modality]
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        # Skip macOS resource-fork sidecars (e.g., ._I01_01_01.png).
        if path.name.startswith("._"):
            continue
        if path.suffix.lower() in valid_ext:
            yield path


def _build_manifest(
    dataset_name: str,
    modality: str,
    imported_root: Path,
    out_manifest_path: Path,
    max_files: int,
):
    out_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "modality", "file_index", "relative_path", "absolute_path"],
        )
        writer.writeheader()
        for index, file_path in enumerate(sorted(_iter_matching_files(imported_root, modality))):
            if max_files > 0 and index >= max_files:
                break
            writer.writerow(
                {
                    "dataset": dataset_name,
                    "modality": modality,
                    "file_index": index,
                    "relative_path": str(file_path.relative_to(imported_root)),
                    "absolute_path": str(file_path.resolve()),
                }
            )


def main():
    args = parse_args()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    imported_summary = []
    for mapping in args.dataset_path:
        dataset_name, source_path = _parse_dataset_mapping(mapping)
        modality = _infer_modality(dataset_name, source_path)

        modality_root = out_root / modality
        imported_dataset_path = modality_root / dataset_name
        if imported_dataset_path.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"Imported dataset path already exists: {imported_dataset_path}. "
                    "Use --overwrite to replace."
                )
            if imported_dataset_path.is_symlink() or imported_dataset_path.is_file():
                imported_dataset_path.unlink()
            else:
                shutil.rmtree(imported_dataset_path)

        modality_root.mkdir(parents=True, exist_ok=True)
        _link_or_copy_dataset(source_path, imported_dataset_path, args.mode)

        manifest_path = out_root / "manifests" / f"{modality}_{dataset_name}.csv"
        _build_manifest(
            dataset_name=dataset_name,
            modality=modality,
            imported_root=imported_dataset_path,
            out_manifest_path=manifest_path,
            max_files=args.manifest_max_files_per_dataset,
        )
        imported_summary.append((dataset_name, modality, imported_dataset_path, manifest_path))

    print(f"[INFO] Imported {len(imported_summary)} dataset(s) into {out_root}")
    for dataset_name, modality, imported_path, manifest_path in imported_summary:
        print(
            f"[INFO] dataset={dataset_name} modality={modality} imported_path={imported_path} manifest={manifest_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
