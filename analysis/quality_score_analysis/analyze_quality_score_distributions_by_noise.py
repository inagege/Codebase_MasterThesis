import argparse
import os
import re
import sys
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path("/tmp") / f"mplconfig_{os.getuid()}"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

import matplotlib
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from result_plot_parsing import parse_split_metadata

QUALITY_FILE_GLOB = "quality_scores_*.csv"
QUALITY_FILE_RE = re.compile(
    r"^quality_scores_(?P<modalities>[a-z]+)_noise_(?P<noise_modalities>[a-z]*)\.csv$",
    re.IGNORECASE,
)
QUALITY_ROOT_PREFIXES = {"calibration", "qwen_scored", "scored", "test"}
MODALITY_ORDER = {"audio": 0, "text": 1, "image": 2, "video": 3}
MODALITY_LETTER_TO_NAME = {"a": "audio", "t": "text", "i": "image", "v": "video"}
MODALITY_NAME_TO_LETTER = {value: key for key, value in MODALITY_LETTER_TO_NAME.items()}
ALLOWED_DATASETS = ["emotion", "homeprice", "imdb", "marine", "nejm", "sentiment", "voxceleb"]
ALLOWED_DATASET_SET = set(ALLOWED_DATASETS)
MAX_POINTS_PER_BOX = 3000
PLOT_DPI = 180


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze raw quality-score distributions by perturbation/severity and write "
            "plots per dataset and per modality."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model folder under out/, e.g. Qwen_7B.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory where plot PNG files are written.",
    )
    return parser.parse_args()


def discover_quality_score_files(root: Path, allowed_dataset_roots: set[str] | None = None) -> list[tuple[str, Path]]:
    if not root.exists():
        return []

    files: list[tuple[str, Path]] = []
    for path in root.rglob(QUALITY_FILE_GLOB):
        relpath = path.relative_to(root).as_posix()
        if allowed_dataset_roots is not None:
            rel_obj = Path(relpath)
            dataset_candidate = rel_obj.parent.name.strip().lower()
            if not dataset_candidate or dataset_candidate not in allowed_dataset_roots:
                continue
        files.append((relpath, path))
    return sorted(files, key=lambda item: item[0])


def infer_dataset_from_relpath(relpath: str) -> str:
    rel_obj = Path(relpath)
    parent_name = rel_obj.parent.name.strip().lower()
    if parent_name:
        return parent_name
    parts = rel_obj.parts
    if not parts:
        return "unknown"

    for idx, token in enumerate(parts[:-1]):
        if token.lower() in QUALITY_ROOT_PREFIXES and idx + 1 < len(parts):
            return parts[idx + 1].lower()

    return str(parts[0]).lower()


def parse_quality_filename(path: Path) -> dict[str, str]:
    match = QUALITY_FILE_RE.match(path.name)
    if not match:
        return {"modalities": "", "noise_modalities": ""}
    return {
        "modalities": (match.group("modalities") or "").lower(),
        "noise_modalities": (match.group("noise_modalities") or "").lower(),
    }


def detect_raw_score_columns(columns: list[str]) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for column in columns:
        if column.endswith("_raw_quality"):
            modality = column[: -len("_raw_quality")]
            found.append((modality, column))
    return found


def _clean_dataset_series(dataset_series: pd.Series, fallback_dataset: str) -> pd.Series:
    cleaned = dataset_series.astype(str).str.strip().str.lower()
    cleaned = cleaned.where(cleaned != "", fallback_dataset)
    cleaned = cleaned.where(cleaned != "nan", fallback_dataset)
    return cleaned


def _noise_level_label(is_unmodified: bool, severity: float | int | None) -> str:
    if is_unmodified:
        return "unmodified"
    if pd.isna(severity):
        return "S=unknown"
    return f"S={int(severity)}"


def _extract_modality_letters(raw_token: object) -> set[str]:
    token = str(raw_token or "").strip().lower()
    if not token:
        return set()

    letters: set[str] = set()
    if "+" in token or "," in token:
        parts = [part.strip() for part in re.split(r"[+,]", token) if part.strip()]
        for part in parts:
            if part in MODALITY_LETTER_TO_NAME:
                letters.add(part)
            elif part in MODALITY_NAME_TO_LETTER:
                letters.add(MODALITY_NAME_TO_LETTER[part])
    else:
        if token in MODALITY_NAME_TO_LETTER:
            letters.add(MODALITY_NAME_TO_LETTER[token])
        for character in token:
            if character in MODALITY_LETTER_TO_NAME:
                letters.add(character)

    return letters


def to_long_quality_scores(csv_path: Path, relative_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "split" not in df.columns:
        return pd.DataFrame()

    score_columns = detect_raw_score_columns(df.columns.tolist())
    if not score_columns:
        return pd.DataFrame()

    rel_obj = Path(relative_path)
    dataset_from_folder = rel_obj.parent.name.strip().lower()
    fallback_dataset = infer_dataset_from_relpath(relative_path)
    if dataset_from_folder in ALLOWED_DATASET_SET:
        dataset_values = pd.Series([dataset_from_folder] * len(df), index=df.index, dtype="string")
    elif "dataset" in df.columns:
        dataset_values = _clean_dataset_series(df["dataset"], fallback_dataset)
    else:
        dataset_values = pd.Series([fallback_dataset] * len(df), index=df.index, dtype="string")

    split_values = df["split"].fillna("").astype(str).str.strip()
    split_meta_by_value = {split_name: parse_split_metadata(split_name) for split_name in split_values.unique()}

    is_unmodified = split_values.map(
        lambda split_name: bool(split_meta_by_value[split_name].get("is_unmodified", False))
    )
    severity = pd.to_numeric(
        split_values.map(lambda split_name: split_meta_by_value[split_name].get("severity")),
        errors="coerce",
    )
    noise_level = [
        _noise_level_label(is_unmodified=unmodified, severity=severity_level)
        for unmodified, severity_level in zip(is_unmodified.tolist(), severity.tolist())
    ]

    perturbation_method = split_values.map(
        lambda split_name: str(split_meta_by_value[split_name].get("perturbation_method", "unknown")).lower()
    )
    perturbation_target = split_values.map(
        lambda split_name: str(split_meta_by_value[split_name].get("perturbation_target", "")).lower()
    )

    quality_file_meta = parse_quality_filename(csv_path)
    file_modalities = _extract_modality_letters(quality_file_meta["modalities"])
    file_noise_modalities = _extract_modality_letters(quality_file_meta["noise_modalities"])
    split_target_letters = perturbation_target.map(_extract_modality_letters)

    allowed_letters_per_row: list[set[str]] = []
    for unmodified, target_letters in zip(is_unmodified.tolist(), split_target_letters.tolist()):
        if unmodified:
            allowed_letters = file_noise_modalities or file_modalities
        else:
            allowed_letters = target_letters or file_noise_modalities or file_modalities
        allowed_letters_per_row.append(set(allowed_letters))

    records: list[pd.DataFrame] = []
    for scored_modality, score_column in score_columns:
        numeric_scores = pd.to_numeric(df[score_column], errors="coerce")
        modality_letter = MODALITY_NAME_TO_LETTER.get(scored_modality.lower(), "")
        if modality_letter:
            modality_allowed_mask = pd.Series(
                [modality_letter in allowed for allowed in allowed_letters_per_row],
                index=df.index,
                dtype=bool,
            )
        else:
            modality_allowed_mask = pd.Series([True] * len(df), index=df.index, dtype=bool)

        valid = numeric_scores.notna() & modality_allowed_mask
        if not valid.any():
            continue

        records.append(
            pd.DataFrame(
                {
                    "source_relpath": relative_path,
                    "dataset": dataset_values[valid].values,
                    "split": split_values[valid].values,
                    "scored_modality": scored_modality,
                    "quality_score": numeric_scores[valid].values.astype(float),
                    "is_unmodified": is_unmodified[valid].values.astype(bool),
                    "severity": severity[valid].values,
                    "noise_level": np.asarray(noise_level, dtype=object)[valid.values],
                    "perturbation_method": perturbation_method[valid].values,
                }
            )
        )

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def _noise_level_sort_value(noise_level: str) -> int:
    if noise_level == "unmodified":
        return -1
    if noise_level.startswith("S="):
        try:
            return int(noise_level[2:])
        except ValueError:
            return 10**9
    return 10**9 + 1


def _method_sort_value(method: str) -> tuple[int, str]:
    normalized = str(method).strip().lower()
    return (0 if normalized == "unmodified" else 1, normalized)


def _ordered_modalities(series: pd.Series) -> list[str]:
    values = [str(value) for value in series.dropna().astype(str).unique().tolist()]
    return sorted(values, key=lambda value: (MODALITY_ORDER.get(value, 99), value))


def _ordered_datasets(series: pd.Series) -> list[str]:
    values = [str(value) for value in series.dropna().astype(str).unique().tolist()]
    rank = {dataset: idx for idx, dataset in enumerate(ALLOWED_DATASETS)}
    return sorted(values, key=lambda value: (rank.get(value, 10**6), value))


def _ordered_methods(series: pd.Series) -> list[str]:
    values = [str(value) for value in series.dropna().astype(str).unique().tolist()]
    return sorted(values, key=_method_sort_value)


def _ordered_perturbation_pairs(data: pd.DataFrame) -> list[tuple[str, str]]:
    if data.empty:
        return []

    pairs_df = (
        data[["perturbation_method", "noise_level"]]
        .dropna()
        .astype(str)
        .drop_duplicates()
    )
    if pairs_df.empty:
        return []

    pairs: list[tuple[str, str]] = []
    methods = sorted(pairs_df["perturbation_method"].unique().tolist(), key=_method_sort_value)

    for method in methods:
        method_rows = pairs_df[pairs_df["perturbation_method"] == method]
        levels = sorted(
            method_rows["noise_level"].unique().tolist(),
            key=lambda level: (_noise_level_sort_value(level), level),
        )
        for level in levels:
            pairs.append((method, level))

    return pairs


def _safe_slug(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "all"


def _downsample(values: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if values.size <= max_points:
        return values
    indices = rng.choice(values.size, size=max_points, replace=False)
    return values[indices]


def _pair_label(method: str, noise_level: str) -> str:
    if method == "unmodified":
        return "unmodified"
    return f"{method}\n{noise_level}"


def _method_label(method: str) -> str:
    normalized = str(method).strip().lower()
    if normalized == "unmodified":
        return "unmodified"
    return normalized.replace("_", " ")


def _color_map_for_labels(labels: list[str]) -> dict[str, tuple]:
    if not labels:
        return {}
    cmap_name = "tab10" if len(labels) <= 10 else "tab20"
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i % cmap.N) for i in range(len(labels))]
    return dict(zip(labels, colors))


def _severity_color_map(levels: list[str]) -> dict[str, tuple]:
    if not levels:
        return {}
    ordered_levels = sorted(levels, key=lambda level: (_noise_level_sort_value(level), level))
    non_unmodified = [level for level in ordered_levels if level != "unmodified"]
    colors: dict[str, tuple] = {}
    if "unmodified" in ordered_levels:
        colors["unmodified"] = (0.35, 0.35, 0.35, 1.0)
    if non_unmodified:
        cmap_name = "tab10" if len(non_unmodified) <= 10 else "tab20"
        cmap = plt.get_cmap(cmap_name)
        for idx, level in enumerate(non_unmodified):
            colors[level] = cmap(idx % cmap.N)
    return colors


def _plot_method_by_severity_boxplot(
    ax,
    data: pd.DataFrame,
    method_order: list[str],
    severity_order: list[str],
    severity_colors: dict[str, tuple],
    rng: np.random.Generator,
) -> bool:
    if not method_order or not severity_order:
        return False

    n_levels = len(severity_order)
    centers = np.arange(len(method_order), dtype=float)
    cluster_width = 0.8
    if n_levels == 1:
        offsets = np.array([0.0])
        box_width = 0.55
    else:
        step = cluster_width / n_levels
        offsets = (np.arange(n_levels, dtype=float) - (n_levels - 1) / 2.0) * step
        box_width = max(0.06, min(0.45, 0.8 * step))

    drew_any = False
    for method_idx, method in enumerate(method_order):
        method_rows = data[data["perturbation_method"].astype(str) == method]
        if method_rows.empty:
            continue

        for level_idx, level in enumerate(severity_order):
            values = method_rows[method_rows["noise_level"].astype(str) == level]["quality_score"].to_numpy(dtype=float)
            if values.size == 0:
                continue

            sampled = _downsample(values, max_points=MAX_POINTS_PER_BOX, rng=rng)
            position = centers[method_idx] + offsets[level_idx]
            boxplot_obj = ax.boxplot(
                [sampled],
                positions=[position],
                widths=box_width,
                patch_artist=True,
                showfliers=False,
            )
            for patch in boxplot_obj["boxes"]:
                patch.set(facecolor=severity_colors[level], edgecolor="#1f1f1f", linewidth=0.8)
            for median in boxplot_obj["medians"]:
                median.set(color="#aa2e25", linewidth=1.2)

            drew_any = True

    if not drew_any:
        return False

    ax.set_xticks(centers)
    ax.set_xticklabels([_method_label(method) for method in method_order], rotation=40, ha="right")
    ax.grid(alpha=0.2, axis="y")
    ax.set_xlim(-0.8, max(0.8, len(method_order) - 0.2))
    return True


def _plot_grouped_boxplot(
    ax,
    data: pd.DataFrame,
    pair_order: list[tuple[str, str]],
    hue_order: list[str],
    hue_column: str,
    hue_colors: dict[str, tuple],
    rng: np.random.Generator,
) -> bool:
    if not pair_order or not hue_order:
        return False

    n_hues = len(hue_order)
    centers = np.arange(len(pair_order), dtype=float)
    cluster_width = 0.8
    if n_hues == 1:
        offsets = np.array([0.0])
        box_width = 0.55
    else:
        step = cluster_width / n_hues
        offsets = (np.arange(n_hues, dtype=float) - (n_hues - 1) / 2.0) * step
        box_width = max(0.06, min(0.45, 0.8 * step))

    drew_any = False
    visible_hues: set[str] = set()

    for pair_idx, (method, noise_level) in enumerate(pair_order):
        pair_rows = data[
            (data["perturbation_method"].astype(str) == method)
            & (data["noise_level"].astype(str) == noise_level)
        ]
        if pair_rows.empty:
            continue

        for hue_idx, hue_value in enumerate(hue_order):
            values = pair_rows[pair_rows[hue_column].astype(str) == hue_value]["quality_score"].to_numpy(dtype=float)
            if values.size == 0:
                continue

            sampled = _downsample(values, max_points=MAX_POINTS_PER_BOX, rng=rng)
            position = centers[pair_idx] + offsets[hue_idx]
            boxplot_obj = ax.boxplot(
                [sampled],
                positions=[position],
                widths=box_width,
                patch_artist=True,
                showfliers=False,
            )
            for patch in boxplot_obj["boxes"]:
                patch.set(facecolor=hue_colors[hue_value], edgecolor="#1f1f1f", linewidth=0.8)
            for median in boxplot_obj["medians"]:
                median.set(color="#aa2e25", linewidth=1.2)

            drew_any = True
            visible_hues.add(hue_value)

    if not drew_any:
        return False

    ax.set_xticks(centers)
    ax.set_xticklabels([_pair_label(method, noise_level) for method, noise_level in pair_order], rotation=40, ha="right")
    ax.grid(alpha=0.2, axis="y")
    ax.set_xlim(-0.8, max(0.8, len(pair_order) - 0.2))

    handles = [mpatches.Patch(facecolor=hue_colors[hue], edgecolor="#1f1f1f", label=hue) for hue in hue_order if hue in visible_hues]
    if handles:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, frameon=True)
    return True


def plot_quality_by_dataset(long_scores: pd.DataFrame, out_dir: Path) -> int:
    created = 0
    rng = np.random.default_rng(42)
    datasets = _ordered_datasets(long_scores["dataset"])

    for dataset in datasets:
        dataset_rows = long_scores[long_scores["dataset"].astype(str) == dataset]
        if dataset_rows.empty:
            continue

        pair_order = _ordered_perturbation_pairs(dataset_rows)
        modality_order = _ordered_modalities(dataset_rows["scored_modality"])
        if not pair_order or not modality_order:
            continue

        hue_colors = _color_map_for_labels(modality_order)
        fig_width = max(11.0, 0.56 * len(pair_order) + 2.8)
        fig, ax = plt.subplots(figsize=(fig_width, 6.0))

        drew = _plot_grouped_boxplot(
            ax=ax,
            data=dataset_rows,
            pair_order=pair_order,
            hue_order=modality_order,
            hue_column="scored_modality",
            hue_colors=hue_colors,
            rng=rng,
        )
        if not drew:
            plt.close(fig)
            continue

        ax.set_title(f"{dataset}: quality score distribution by perturbation and severity")
        ax.set_xlabel("perturbation + severity")
        ax.set_ylabel("quality score")
        fig.tight_layout(rect=(0, 0, 0.84, 1))

        out_path = out_dir / f"quality_{_safe_slug(dataset)}.png"
        fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        created += 1

    return created


def plot_quality_by_modality(long_scores: pd.DataFrame, out_dir: Path) -> int:
    created = 0
    rng = np.random.default_rng(43)
    modalities = _ordered_modalities(long_scores["scored_modality"])

    for modality in modalities:
        modality_rows = long_scores[long_scores["scored_modality"].astype(str) == modality]
        if modality_rows.empty:
            continue

        dataset_order = _ordered_datasets(modality_rows["dataset"])
        method_order = _ordered_methods(modality_rows["perturbation_method"])
        severity_order = sorted(
            [str(value) for value in modality_rows["noise_level"].dropna().astype(str).unique().tolist()],
            key=lambda level: (_noise_level_sort_value(level), level),
        )
        if not method_order or not dataset_order or not severity_order:
            continue

        severity_colors = _severity_color_map(severity_order)

        subplot_width = max(3.6, 0.46 * len(method_order) + 1.9)
        fig_width = max(11.0, subplot_width * len(dataset_order))
        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(dataset_order),
            figsize=(fig_width, 6.2),
            sharey=True,
        )

        if len(dataset_order) == 1:
            axes = np.asarray([axes], dtype=object)

        drew_any = False
        for idx, dataset in enumerate(dataset_order):
            ax = axes[idx]
            dataset_rows = modality_rows[modality_rows["dataset"].astype(str) == dataset]
            drew = _plot_method_by_severity_boxplot(
                ax=ax,
                data=dataset_rows,
                method_order=method_order,
                severity_order=severity_order,
                severity_colors=severity_colors,
                rng=rng,
            )
            if drew:
                drew_any = True
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="#666666")
                ax.set_xticks(np.arange(len(method_order), dtype=float))
                ax.set_xticklabels([_method_label(method) for method in method_order], rotation=40, ha="right")
                ax.set_xlim(-0.8, max(0.8, len(method_order) - 0.2))
                ax.grid(alpha=0.2, axis="y")

            ax.set_title(dataset)
            ax.set_xlabel("perturbation")
            if idx == 0:
                ax.set_ylabel("quality score")
            else:
                ax.set_ylabel("")

        if not drew_any:
            plt.close(fig)
            continue

        handles = [
            mpatches.Patch(facecolor=severity_colors[level], edgecolor="#1f1f1f", label=level)
            for level in severity_order
            if level in severity_colors
        ]
        if handles:
            fig.legend(
                handles=handles,
                loc="upper center",
                ncol=min(len(handles), 6),
                frameon=True,
                bbox_to_anchor=(0.5, 0.99),
            )

        fig.suptitle(f"{modality}: quality score distribution by perturbation (colored by severity)")
        fig.tight_layout(rect=(0, 0, 1, 0.92))

        out_path = out_dir / f"quality_{_safe_slug(modality)}.png"
        fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        created += 1

    return created


def _clean_generated_plots(plot_dir: Path) -> None:
    if not plot_dir.exists():
        return
    for path in plot_dir.glob("quality_*.png"):
        if path.is_file():
            path.unlink()


def main() -> None:
    args = parse_args()
    quality_root = Path("../out") / args.model / "qwen_scored"
    if not quality_root.exists():
        raise FileNotFoundError(f"Input root not found: {quality_root}")

    output_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("analysis") / "out" / args.model / "plots" / "quality_score_distribution_by_noise"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    quality_files = discover_quality_score_files(root=quality_root, allowed_dataset_roots=ALLOWED_DATASET_SET)
    if not quality_files:
        raise FileNotFoundError(f"No files matching {QUALITY_FILE_GLOB!r} found under {quality_root}")

    long_frames = []
    for relpath, csv_path in quality_files:
        long_frame = to_long_quality_scores(csv_path=csv_path, relative_path=relpath)
        if not long_frame.empty:
            long_frames.append(long_frame)

    if not long_frames:
        raise ValueError("No usable raw quality score columns were found in discovered CSV files.")

    long_scores = pd.concat(long_frames, ignore_index=True)
    long_scores = long_scores[long_scores["dataset"].astype(str).isin(ALLOWED_DATASET_SET)].copy()
    if long_scores.empty:
        raise ValueError(
            "No rows left after dataset filtering. Allowed datasets: "
            + ", ".join(ALLOWED_DATASETS)
        )
    _clean_generated_plots(output_dir)

    per_modality_count = plot_quality_by_modality(long_scores=long_scores, out_dir=output_dir)

    print(f"[INFO] Input root: {quality_root}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Dataset filter: {', '.join(ALLOWED_DATASETS)}")
    print(f"[INFO] Analyzed {len(long_scores)} quality-score rows from {len(quality_files)} files.")
    print(f"[INFO] Generated {per_modality_count} modality plots (quality_<modality>.png).")


if __name__ == "__main__":
    main()
