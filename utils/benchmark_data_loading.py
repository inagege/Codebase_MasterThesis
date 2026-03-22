from __future__ import annotations

import hashlib
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

ALL_MODALITIES = {"text", "audio", "video", "image"}
DATASET_MODALITIES = {
    "meld": {"text", "audio", "video"},
    "homeprice": {"text", "image"},
    "imdb": {"text", "image"},
    "voxceleb": {"audio", "video"},
    "nejm": {"text", "image"},
    "marine": {"audio", "image"},
}

MELD_SPLIT_ROOT = {
    "train": "data/MELD.Raw/train_splits",
    "test": "data/MELD.Raw/output_repeated_splits_test",
    "val": "data/MELD.Raw/dev_splits_complete",
}
MELD_SPLIT_META = {
    "train": "train_sent_emo.csv",
    "test": "test_sent_emo.csv",
    "val": "dev_sent_emo.csv",
}


def normalize_dataset_name(dataset: str) -> str:
    key = (dataset or "").strip().lower()
    aliases = {
        "meld": "meld",
        "meld.raw": "meld",
        "homeprice": "homeprice",
        "home_price": "homeprice",
        "imdb": "imdb",
        "voxceleb": "voxceleb",
        "voxceleb2": "voxceleb",
        "nejm": "nejm",
        "marine": "marine",
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Valid: {sorted(set(aliases.values()))}"
        )
    return aliases[key]


def default_modalities_for_dataset(dataset: str) -> set[str]:
    return set(DATASET_MODALITIES[dataset])


def normalize_modalities(mod_str: str | None):
    if mod_str is None:
        return None
    mods = {m.strip().lower() for m in mod_str.split(",") if m.strip()}
    bad = mods - ALL_MODALITIES
    if bad:
        raise ValueError(f"Unknown modalities: {sorted(bad)}. Valid: {sorted(ALL_MODALITIES)}")
    if not mods:
        raise ValueError("No modalities selected.")
    return mods


def normalize_splits(split_str: str):
    splits = {s.strip().lower() for s in split_str.split(",") if s.strip()}
    if not splits:
        raise ValueError("No split selected.")
    return splits


def normalize_meld_task(task: str | None):
    task = (task or "sentiment").strip().lower()
    if task not in {"sentiment", "emotion"}:
        raise ValueError("For MELD, use --classification-task sentiment or emotion.")
    return task, task.capitalize()


def normalize_voxceleb_label_column(task: str | None):
    token = (task or "nationality").strip().lower()
    mapping = {
        "name": "Name",
        "nationality": "nationality_wiki",
        "nationality_wiki": "nationality_wiki",
    }
    if token not in mapping:
        raise ValueError(
            "For VoxCeleb, choose label as --classification-task name or --classification-task nationality."
        )
    return mapping[token]


def get_prompt_for_classification(dataset: str, meld_task: str | None):
    if dataset == "meld":
        if meld_task == "sentiment":
            return (
                "The dataset contains utterances from the Friends TV series. "
                "Classify the sample sentiment by answering with exactly one word: neutral, negative, or positive."
            )
        return (
            "The dataset contains utterances from the Friends TV series. "
            "Classify the sample emotion by answering with exactly one word: anger, disgust, sadness, joy, neutral, surprise, or fear."
        )
    if dataset == "homeprice":
        return (
            "The dataset contains information on homes in Austin. "
            "Predict the price bin and answer with exactly the number (0, 1, 2, 3 or 4) of the price bin you think the given sample is on."
            "The options are 0: 5500$-205000$, 1: 205001$-325000$, 2: 325000$-525000$, 3: 525001$-1100000$, 4: 1100001$-13500000$"
        )
    if dataset == "imdb":
        return (
            "The dataset contains information about movies."
            "Classify the movie genre and answer with exactly one genre label: action, horror, comedy or romance."
        )
    if dataset == "voxceleb":
        target = "speaker nationality"
        nationalities = "Germany, Canada, The-United-Kingdom-1, India, Australia,The-United-States-of-America, " \
        "Honduras, The-Republic-of-Ireland, Jordan-215, Switzerland, Kingdom-of-the-Netherlands, Austria, Spain, " \
        "Slovenia, Belgium, Israel, Serbia, Panama, France, Cuba, Morocco, Soviet-Union-1, Sweden, Denmark, Norway, " \
        "Russia, Chile, Greece, Georgia-country, Brazil, Malaysia, Italy, Mexico, Croatia, Indonesia, The-Czech-Republic-1, " \
        "Jamaica, Turkey-country, Poland, Portugal, Iran, China, Yugoslavia, Puerto-Rico, Japan, South-Africa, Sri-Lanka, Colombia, " \
        "Bosnia-and-Herzegovina, Uruguay, Angola, New-Zealand, Hong-Kong, Finland, South-Korea, Luxembourg, The-Philippines-1, " \
        "British-Colonial-Rule, Taiwan, Malta, Haiti, Dominican-Republic, German-Democratic-Republic-DDR, Monaco, Armenia, Slovakia-1, " \
        "Czechoslovakia-former-country, Hungary, Scotland, Kosovo, Argentina, Romania, Peru, Egypt, Lithuania, Pakistan, Bangladesh, " \
        "Vietnam, Nepal, Ethiopia, Senegal, Cambodia, Lebanon, Venezuela, Benin, Saudi-Arabia, England, Algeria, Nicaragua, Nigeria, " \
        "Tanzania, Ghana, Burundi, Guatemala, Trinidad-and-Tobago, Bulgaria, Thailand, Weimar-Republic, Albania, Zimbabwe, South-Sudan, " \
        "Iraq, The-Bahamas, Montenegro, Cameroon, Belarus-1, Uganda, Paraguay, Fiji, Syria, Ivory-Coast, Northern-Ireland-1, Estonia, " \
        "Singapore, Ukraine, State-of-Palestine, Saint-Lucia, Mali, Republic-of-North-Macedonia, Ecuador, British-Virgin-Islands, " \
        "El-Salvador, Democratic-Republic-of-the-Congo-1, Samoa, Guinea, Kenya, Costa-Rica-1, Afghanistan, Colonial-Hong-Kong, " \
        "Latvia, Togo, West-Germany, Palestine, Sierra-Leone, Bolivia, Moldova-1, Burkina-Faso, Tunisia, Azerbaijan, Malawi, Iceland"
        return (
            f"The dataset contains audio and or video of speakers. "
            f"Predict the {target} and answer with exactly one of the following labels: {nationalities}"
        )
    if dataset == "nejm":
        return (
            "The dataset contains medical questions and images. "
            "Classify the disease using the per-sample options. "
            "Answer with exactly one disease label and no explanation."
        )
    if dataset == "marine":
        return (
            "The dataset contains information on marine species."
            "Classify the species and answer with exactly one of the following species labels:"
            "Atlantic Spotted Dolphin, Bearded Seal, Beluga White Whale, Bottlenose Dolphin, Bowhead Whale, "
            "Clymene Dolphin, Common Dolphin, False Killer Whale, Fin Finback Whale, Frasers Dolphin, Grampus Rissos Dolphin, "
            "Harp Seal, Humpback Whale, Killer Whale, Leopard Seal, Long-Finned Pilot Whale, Melon Headed Whale, Minke Whale, "
            "Narwhal, Northern Right Whale, Pantropical Spotted Dolphin, Ross Seal, Rough-Toothed Dolphin, Short-Finned Pacific Pilot Whale, "
            "Southern Right Whale, Sperm Whale, Spinner Dolphin, Striped Dolphin, Walrus, Weddell Seal, White-beaked Dolphin, "
            "White-sided Dolphin"
        )
    raise ValueError(f"Unhandled dataset: {dataset}")


def validate_modalities(dataset: str, enabled_modalities, noisy_modalities):
    allowed = DATASET_MODALITIES[dataset]
    unsupported = enabled_modalities - allowed
    if unsupported:
        raise ValueError(
            f"Modalities {sorted(unsupported)} are not available for dataset {dataset}. "
            f"Allowed: {sorted(allowed)}"
        )
    if noisy_modalities is not None:
        unknown_noisy = noisy_modalities - allowed
        if unknown_noisy:
            raise ValueError(
                f"Unknown noisy modalities for dataset {dataset}: {sorted(unknown_noisy)}"
            )
        if not noisy_modalities.issubset(enabled_modalities):
            raise ValueError("--noisy-modalities must be a subset of --modalities.")


def load_samples(dataset, args, enabled_modalities, noisy_modalities, label_column):
    noise_severity = getattr(args, "noise_severity", None)

    if dataset == "meld":
        return _load_meld_samples(args, noisy_modalities, label_column)
    if dataset == "homeprice":
        return _load_homeprice_samples(enabled_modalities, noisy_modalities, noise_severity=noise_severity)
    if dataset == "imdb":
        return _load_imdb_samples(enabled_modalities, noisy_modalities, noise_severity=noise_severity)
    if dataset == "voxceleb":
        return _load_voxceleb_samples(
            args,
            enabled_modalities,
            noisy_modalities,
            label_column,
            noise_severity=noise_severity,
        )
    if dataset == "nejm":
        return _load_nejm_samples(enabled_modalities, noisy_modalities, noise_severity=noise_severity)
    if dataset == "marine":
        return _load_marine_samples(enabled_modalities, noisy_modalities, noise_severity=noise_severity)
    raise ValueError(f"Unsupported dataset {dataset}")


def _label_for_sample(sample) -> str:
    return _sanitize_value(sample.get("label")) or "unknown"


def _sample_selection_key(sample) -> str:
    return _sanitize_value(sample.get("sample_id"))


def _deterministic_sample_rank(sample):
    dataset = _sanitize_value(sample.get("dataset"))
    split = _sanitize_value(sample.get("split"))
    sample_id = _sanitize_value(sample.get("sample_id"))
    file_name = _sanitize_value(sample.get("file"))
    label = _label_for_sample(sample)
    token = f"{dataset}|{split}|{sample_id}|{file_name}|{label}"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return digest, split, sample_id, file_name


def select_stratified_samples(samples, sample_limit: int):
    """Deterministically select a label-stratified subset of samples."""
    if sample_limit < 1:
        raise ValueError("sample_limit must be >= 1")
    if sample_limit >= len(samples):
        return samples

    grouped = defaultdict(list)
    for sample in samples:
        grouped[_label_for_sample(sample)].append(sample)

    labels = sorted(grouped)
    for label in labels:
        grouped[label] = sorted(grouped[label], key=_deterministic_sample_rank)

    total_count = len(samples)
    per_label_take = {}
    remainder_scores = []
    allocated = 0
    for label in labels:
        group_size = len(grouped[label])
        exact_target = sample_limit * group_size / total_count
        base_take = int(exact_target)
        per_label_take[label] = min(base_take, group_size)
        allocated += per_label_take[label]
        remainder_scores.append((exact_target - base_take, label))

    remaining = sample_limit - allocated
    remainder_scores.sort(key=lambda item: (-item[0], item[1]))
    for _, label in remainder_scores:
        if remaining == 0:
            break
        if per_label_take[label] < len(grouped[label]):
            per_label_take[label] += 1
            remaining -= 1

    if remaining > 0:
        leftovers = []
        for label in labels:
            leftovers.extend(grouped[label][per_label_take[label] :])
        leftovers.sort(key=_deterministic_sample_rank)
        for sample in leftovers[:remaining]:
            per_label_take[_label_for_sample(sample)] += 1

    selected = []
    for label in labels:
        selected.extend(grouped[label][: per_label_take[label]])

    selected.sort(key=_deterministic_sample_rank)
    return selected[:sample_limit]


def filter_samples_by_sample_id(samples, selected_sample_ids):
    normalized_ids = {_sanitize_value(sample_id) for sample_id in selected_sample_ids if _sanitize_value(sample_id)}
    return [sample for sample in samples if _sample_selection_key(sample) in normalized_ids]


def _sanitize_value(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _parse_meld_ids(filename: str):
    match = re.search(r"\bdia(\d+)_utt(\d+)\b", filename, flags=re.IGNORECASE)
    if not match:
        return -1, -1
    return int(match.group(1)), int(match.group(2))


def _resolve_media_path_with_fallback(base_path: Path):
    if base_path.exists():
        return base_path
    candidates = [
        base_path.with_suffix(".jpg"),
        base_path.with_suffix(".jpeg"),
        base_path.with_suffix(".png"),
        base_path.with_suffix(".wav"),
        base_path.with_suffix(".mp4"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _variant_matches_noise_severity(variant_name: str, noise_severity: int) -> bool:
    return re.search(
        rf"(?:^|[_-])S={re.escape(str(noise_severity))}(?:[_-]|$)",
        variant_name,
    ) is not None


def _iter_noise_variants(noise_root: Path, modality: str, noise_severity: int | None = None):
    modality_root = noise_root / modality
    if not modality_root.exists():
        print(f"[WARN] No noisy root found for modality {modality}: {modality_root}", flush=True)
        return []

    variants = sorted(path for path in modality_root.iterdir() if path.is_dir())
    if noise_severity is None:
        return variants

    filtered_variants = [
        path for path in variants if _variant_matches_noise_severity(path.name, noise_severity)
    ]
    if not filtered_variants:
        print(
            "[WARN] No noisy variants found for modality "
            f"{modality} at severity S={noise_severity} under {modality_root}",
            flush=True,
        )
    return filtered_variants


def _load_text_variant_map(csv_path: Path, key_column: str, text_column: str):
    if not csv_path.exists():
        print(f"[WARN] No noisy text metadata found: {csv_path}", flush=True)
        return {}

    df = pd.read_csv(csv_path)
    mapping = {}
    for _, row in df.iterrows():
        key = _sanitize_value(row.get(key_column))
        if key:
            mapping[key] = _sanitize_value(row.get(text_column))
    return mapping


def _requested_meld_splits(split_arg: str):
    splits = normalize_splits(split_arg)
    mapped = set()
    for split in splits:
        if split == "dev":
            split = "val"
        if split not in {"train", "val", "test"}:
            raise ValueError("For MELD, use --split train,val,test (any subset).")
        mapped.add(split)
    return sorted(mapped)


def _get_meld_split_configs(split: str, noisy_modalities, noise_severity: int | None = None):
    root = MELD_SPLIT_ROOT[split]
    meta_csv = os.path.join(root, MELD_SPLIT_META[split])
    configs = []

    if noisy_modalities is None:
        configs.append(
            {
                "media_dir": Path(root) / "unmodified",
                "meta_csv": Path(meta_csv),
                "variant": f"{split}_unmodified",
            }
        )
        return configs

    for modality in noisy_modalities - {"text"}:
        for variant in _iter_noise_variants(Path(root), modality, noise_severity):
            configs.append(
                {
                    "media_dir": variant,
                    "meta_csv": Path(meta_csv),
                    "variant": f"{split}_{variant.name}",
                }
            )

    if "text" in noisy_modalities:
        for variant in _iter_noise_variants(Path(root), "text", noise_severity):
            configs.append(
                {
                    "media_dir": Path(root) / "unmodified",
                    "meta_csv": variant / "metadata.csv",
                    "variant": f"{split}_{variant.name}",
                }
            )

    return configs


def _load_meld_samples(args, noisy_modalities, label_column):
    samples = []
    for split in _requested_meld_splits(args.split):
        split_configs = _get_meld_split_configs(
            split,
            noisy_modalities,
            noise_severity=getattr(args, "noise_severity", None),
        )
        if not split_configs:
            print(f"[WARN] No MELD configs found for split {split}", flush=True)
            continue

        for cfg in split_configs:
            media_dir = cfg["media_dir"]
            meta_csv = cfg["meta_csv"]
            variant = cfg["variant"]
            if not media_dir.exists():
                print(f"[WARN] Media dir does not exist: {media_dir}", flush=True)
                continue
            if not meta_csv.exists():
                print(f"[WARN] Metadata CSV does not exist: {meta_csv}", flush=True)
                continue

            meta_df = pd.read_csv(meta_csv)
            records = {}
            for _, row in meta_df.iterrows():
                try:
                    dia_id = int(row["Dialogue_ID"])
                    utt_id = int(row["Utterance_ID"])
                except Exception:
                    continue
                records[(dia_id, utt_id)] = {
                    "text": _sanitize_value(row.get("Utterance")),
                    "label": _sanitize_value(row.get(label_column, "unknown")) or "unknown",
                }

            mp4_files = sorted(
                path for path in media_dir.iterdir() if path.is_file() and path.suffix.lower() == ".mp4"
            )
            print(f"[INFO] MELD {variant}: found {len(mp4_files)} mp4 files in {media_dir}", flush=True)
            for mp4_path in mp4_files:
                dia_id, utt_id = _parse_meld_ids(mp4_path.name)
                record = records.get((dia_id, utt_id), {})
                samples.append(
                    {
                        "dataset": "meld",
                        "split": variant,
                        "sample_id": mp4_path.stem,
                        "file": mp4_path.name,
                        "text": record.get("text", ""),
                        "audio": mp4_path.parent / args.audio_subdir / f"{mp4_path.stem}.wav",
                        "video": mp4_path,
                        "image": None,
                        "label": record.get("label", "unknown"),
                    }
                )
    return samples


def _resolve_homeprice_csv_path():
    candidates = [
        Path("data/HomePrice/data_price_binned.csv"),
        Path("data/HomePrice/data_prive_binned.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find HomePrice CSV (expected data_price_binned.csv or data_prive_binned.csv)."
    )


def _load_homeprice_samples(enabled_modalities, noisy_modalities, noise_severity: int | None = None):
    csv_path = _resolve_homeprice_csv_path()
    image_dir = Path("data/HomePrice/homeImages")
    noise_root = Path("data/HomePrice/noise")
    df = pd.read_csv(csv_path)

    if noisy_modalities is None:
        samples = []
        for _, row in df.iterrows():
            image_name = _sanitize_value(row.get("homeImage"))
            if not image_name:
                continue
            samples.append(
                {
                    "dataset": "homeprice",
                    "split": "all",
                    "sample_id": image_name,
                    "file": image_name,
                    "text": _sanitize_value(row.get("description")),
                    "audio": None,
                    "video": None,
                    "image": image_dir / image_name,
                    "label": _sanitize_value(row.get("price_bin", "unknown")) or "unknown",
                }
            )
        return samples

    samples = []
    base_rows = [row for _, row in df.iterrows()]

    if "image" in noisy_modalities:
        for variant in _iter_noise_variants(noise_root, "image", noise_severity):
            for row in base_rows:
                image_name = _sanitize_value(row.get("homeImage"))
                if not image_name:
                    continue
                samples.append(
                    {
                        "dataset": "homeprice",
                        "split": f"all_{variant.name}",
                        "sample_id": image_name,
                        "file": image_name,
                        "text": _sanitize_value(row.get("description")),
                        "audio": None,
                        "video": None,
                        "image": variant / image_name,
                        "label": _sanitize_value(row.get("price_bin", "unknown")) or "unknown",
                    }
                )

    if "text" in noisy_modalities:
        for variant in _iter_noise_variants(noise_root, "text", noise_severity):
            text_map = _load_text_variant_map(
                csv_path=variant / csv_path.name,
                key_column="homeImage",
                text_column="description",
            )
            for row in base_rows:
                image_name = _sanitize_value(row.get("homeImage"))
                if not image_name:
                    continue
                samples.append(
                    {
                        "dataset": "homeprice",
                        "split": f"all_{variant.name}",
                        "sample_id": image_name,
                        "file": image_name,
                        "text": text_map.get(image_name, _sanitize_value(row.get("description"))),
                        "audio": None,
                        "video": None,
                        "image": image_dir / image_name,
                        "label": _sanitize_value(row.get("price_bin", "unknown")) or "unknown",
                    }
                )

    return samples


def _load_imdb_samples(enabled_modalities, noisy_modalities, noise_severity: int | None = None):
    csv_path = Path("data/IMDB/IMDB_four_genre_larger_plot_description.csv")
    image_dir = Path("data/IMDB/IMDB_four_genre_posters")
    noise_root = Path("data/IMDB/noise")
    df = pd.read_csv(csv_path)

    if noisy_modalities is None:
        samples = []
        for _, row in df.iterrows():
            movie_id = _sanitize_value(row.get("movie_id"))
            if not movie_id:
                continue
            image_path = _resolve_media_path_with_fallback(image_dir / movie_id)
            samples.append(
                {
                    "dataset": "imdb",
                    "split": "all",
                    "sample_id": movie_id,
                    "file": f"{movie_id}.jpg",
                    "text": _sanitize_value(row.get("description")),
                    "audio": None,
                    "video": None,
                    "image": image_path if image_path is not None else image_dir / f"{movie_id}.jpg",
                    "label": _sanitize_value(row.get("genre", "unknown")) or "unknown",
                }
            )
        return samples

    samples = []
    base_rows = [row for _, row in df.iterrows()]

    if "image" in noisy_modalities:
        for variant in _iter_noise_variants(noise_root, "image", noise_severity):
            for row in base_rows:
                movie_id = _sanitize_value(row.get("movie_id"))
                if not movie_id:
                    continue
                noisy_image_path = _resolve_media_path_with_fallback(variant / movie_id)
                samples.append(
                    {
                        "dataset": "imdb",
                        "split": f"all_{variant.name}",
                        "sample_id": movie_id,
                        "file": f"{movie_id}.jpg",
                        "text": _sanitize_value(row.get("description")),
                        "audio": None,
                        "video": None,
                        "image": noisy_image_path if noisy_image_path is not None else variant / f"{movie_id}.jpg",
                        "label": _sanitize_value(row.get("genre", "unknown")) or "unknown",
                    }
                )

    if "text" in noisy_modalities:
        for variant in _iter_noise_variants(noise_root, "text", noise_severity):
            text_map = _load_text_variant_map(
                csv_path=variant / csv_path.name,
                key_column="movie_id",
                text_column="description",
            )
            for row in base_rows:
                movie_id = _sanitize_value(row.get("movie_id"))
                if not movie_id:
                    continue
                image_path = _resolve_media_path_with_fallback(image_dir / movie_id)
                samples.append(
                    {
                        "dataset": "imdb",
                        "split": f"all_{variant.name}",
                        "sample_id": movie_id,
                        "file": f"{movie_id}.jpg",
                        "text": text_map.get(movie_id, _sanitize_value(row.get("description"))),
                        "audio": None,
                        "video": None,
                        "image": image_path if image_path is not None else image_dir / f"{movie_id}.jpg",
                        "label": _sanitize_value(row.get("genre", "unknown")) or "unknown",
                    }
                )

    return samples


def _parse_nejm_label(answer: str):
    answer = (answer or "").strip()
    if ":" in answer:
        return answer.split(":", 1)[1].strip()
    return answer


def _clean_nejm_option_label(label: str) -> str:
    cleaned = re.sub(r"[\r\n]+", " ", label or "")
    cleaned = cleaned.replace("\\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.strip("[]{}()")
    cleaned = cleaned.strip()
    cleaned = cleaned.strip("'\"")
    cleaned = cleaned.strip(" ,;")
    cleaned = cleaned.strip("'\"")
    cleaned = cleaned.strip(" ,;")
    return cleaned


def _extract_nejm_option_labels(raw_value) -> list[str]:
    text = _sanitize_value(raw_value)
    if not text:
        return []

    marker_matches = list(re.finditer(r"([A-Ea-e])\s*:\s*", text))
    if not marker_matches:
        return []

    labels_by_letter = {}
    for idx, match in enumerate(marker_matches):
        letter = match.group(1).upper()
        if letter in labels_by_letter:
            continue
        start = match.end()
        end = marker_matches[idx + 1].start() if idx + 1 < len(marker_matches) else len(text)
        label = _clean_nejm_option_label(text[start:end])
        if label:
            labels_by_letter[letter] = label

    return [labels_by_letter[letter] for letter in "ABCDE" if letter in labels_by_letter]


def _extract_nejm_options_for_row(row) -> list[str]:
    labels = _extract_nejm_option_labels(row.get("options"))
    if labels:
        return labels
    conversation_text = _sanitize_value(row.get("conversations"))
    if not conversation_text:
        return []

    for assistant_marker in ("{'from': 'gpt'", '{"from": "gpt"'):
        conversation_text = conversation_text.split(assistant_marker, 1)[0]
    return _extract_nejm_option_labels(conversation_text)


def _load_nejm_samples(enabled_modalities, noisy_modalities, noise_severity: int | None = None):
    csv_path = Path("data/NEJM/metadata.csv")
    image_root = Path("data/NEJM/images")
    noise_root = Path("data/NEJM/noise")
    df = pd.read_csv(csv_path)

    if noisy_modalities is None:
        samples = []
        for _, row in df.iterrows():
            image_id = _sanitize_value(row.get("image_id"))
            raw_image_path = _sanitize_value(row.get("image_path"))
            if raw_image_path:
                image_path = Path(raw_image_path)
            else:
                image_path = image_root / f"image_{image_id}.jpg"
            if not image_path.exists():
                fallback = _resolve_media_path_with_fallback(image_path.with_suffix(""))
                if fallback is not None:
                    image_path = fallback

            question = _sanitize_value(row.get("question")).replace("<image>", "").strip()
            label = _parse_nejm_label(_sanitize_value(row.get("answer")))
            option_labels = _extract_nejm_options_for_row(row)
            options_text = " | ".join(option_labels) if option_labels else _sanitize_value(row.get("options"))
            samples.append(
                {
                    "dataset": "nejm",
                    "split": "all",
                    "sample_id": image_id,
                    "file": image_path.name,
                    "text": question,
                    "options": options_text,
                    "option_labels": option_labels,
                    "audio": None,
                    "video": None,
                    "image": image_path,
                    "label": label or "unknown",
                }
            )
        return samples

    base_rows = [row for _, row in df.iterrows()]
    samples = []

    if "image" in noisy_modalities:
        for variant in _iter_noise_variants(noise_root, "image", noise_severity):
            for row in base_rows:
                image_id = _sanitize_value(row.get("image_id"))
                raw_image_path = _sanitize_value(row.get("image_path"))
                if raw_image_path:
                    base_image_path = Path(raw_image_path)
                else:
                    base_image_path = image_root / f"image_{image_id}.jpg"

                noisy_image_path = variant / base_image_path.name
                if not noisy_image_path.exists():
                    fallback = _resolve_media_path_with_fallback((variant / base_image_path.name).with_suffix(""))
                    if fallback is not None:
                        noisy_image_path = fallback

                question = _sanitize_value(row.get("question")).replace("<image>", "").strip()
                label = _parse_nejm_label(_sanitize_value(row.get("answer")))
                option_labels = _extract_nejm_options_for_row(row)
                options_text = " | ".join(option_labels) if option_labels else _sanitize_value(row.get("options"))
                samples.append(
                    {
                        "dataset": "nejm",
                        "split": f"all_{variant.name}",
                        "sample_id": image_id,
                        "file": noisy_image_path.name,
                        "text": question,
                        "options": options_text,
                        "option_labels": option_labels,
                        "audio": None,
                        "video": None,
                        "image": noisy_image_path,
                        "label": label or "unknown",
                    }
                )

    if "text" in noisy_modalities:
        for variant in _iter_noise_variants(noise_root, "text", noise_severity):
            text_map = _load_text_variant_map(
                csv_path=variant / "metadata.csv",
                key_column="image_id",
                text_column="question",
            )
            for row in base_rows:
                image_id = _sanitize_value(row.get("image_id"))
                raw_image_path = _sanitize_value(row.get("image_path"))
                if raw_image_path:
                    image_path = Path(raw_image_path)
                else:
                    image_path = image_root / f"image_{image_id}.jpg"
                if not image_path.exists():
                    fallback = _resolve_media_path_with_fallback(image_path.with_suffix(""))
                    if fallback is not None:
                        image_path = fallback

                question = text_map.get(image_id, _sanitize_value(row.get("question")))
                question = question.replace("<image>", "").strip()
                label = _parse_nejm_label(_sanitize_value(row.get("answer")))
                option_labels = _extract_nejm_options_for_row(row)
                options_text = " | ".join(option_labels) if option_labels else _sanitize_value(row.get("options"))
                samples.append(
                    {
                        "dataset": "nejm",
                        "split": f"all_{variant.name}",
                        "sample_id": image_id,
                        "file": image_path.name,
                        "text": question,
                        "options": options_text,
                        "option_labels": option_labels,
                        "audio": None,
                        "video": None,
                        "image": image_path,
                        "label": label or "unknown",
                    }
                )

    return samples


def _build_voxceleb_samples_for_roots(
    args, enabled_modalities, speaker_to_label, split_name, video_root: Path, audio_root: Path
):
    video_map = {}
    audio_map = {}

    if "video" in enabled_modalities:
        if not video_root.exists():
            print(f"[WARN] VoxCeleb video root does not exist: {video_root}", flush=True)
        for mp4_path in video_root.rglob("*.mp4"):
            if args.audio_subdir in mp4_path.parts:
                continue
            rel = mp4_path.relative_to(video_root).with_suffix("").as_posix()
            video_map[rel] = mp4_path

    if "audio" in enabled_modalities:
        if not audio_root.exists():
            print(f"[WARN] VoxCeleb audio root does not exist: {audio_root}", flush=True)
        for wav_path in audio_root.rglob("*.wav"):
            rel = wav_path.relative_to(audio_root).with_suffix("").as_posix()
            audio_map[rel] = wav_path

    if "audio" in enabled_modalities and "video" in enabled_modalities:
        keys = sorted(set(audio_map) & set(video_map))
    elif "audio" in enabled_modalities:
        keys = sorted(audio_map)
    else:
        keys = sorted(video_map)

    samples = []
    for key in keys:
        speaker_id = key.split("/", 1)[0]
        label = speaker_to_label.get(speaker_id, "unknown")
        reference_path = video_map.get(key) if "video" in enabled_modalities else audio_map.get(key)
        file_name = reference_path.name if reference_path is not None else key
        samples.append(
            {
                "dataset": "voxceleb",
                "split": split_name,
                "sample_id": key,
                "file": file_name,
                "text": "",
                "audio": audio_map.get(key),
                "video": video_map.get(key),
                "image": None,
                "label": label,
            }
        )
    return samples


def _load_voxceleb_samples(args, enabled_modalities, noisy_modalities, label_column, noise_severity: int | None = None):
    base_video_root = Path("data/VoxCeleb2/dev/mp4")
    base_audio_root = base_video_root / args.audio_subdir
    noise_root = Path("data/VoxCeleb2/dev/noise")
    speaker_csv = Path("data/VoxCeleb2/speaker_information.csv")

    speaker_df = pd.read_csv(speaker_csv)
    if label_column not in speaker_df.columns:
        raise ValueError(
            f"Label column {label_column!r} not found in {speaker_csv}. "
            f"Available: {list(speaker_df.columns)}"
        )

    speaker_to_label = {}
    for _, row in speaker_df.iterrows():
        speaker_id = _sanitize_value(row.get("VoxCeleb_ID"))
        if speaker_id:
            speaker_to_label[speaker_id] = _sanitize_value(row.get(label_column)) or "unknown"

    if noisy_modalities is None:
        return _build_voxceleb_samples_for_roots(
            args=args,
            enabled_modalities=enabled_modalities,
            speaker_to_label=speaker_to_label,
            split_name="dev",
            video_root=base_video_root,
            audio_root=base_audio_root,
        )

    samples = []
    if "video" in noisy_modalities:
        for variant in _iter_noise_variants(noise_root, "video", noise_severity):
            samples.extend(
                _build_voxceleb_samples_for_roots(
                    args=args,
                    enabled_modalities=enabled_modalities,
                    speaker_to_label=speaker_to_label,
                    split_name=f"dev_{variant.name}",
                    video_root=variant,
                    audio_root=base_audio_root,
                )
            )
    if "audio" in noisy_modalities:
        for variant in _iter_noise_variants(noise_root, "audio", noise_severity):
            samples.extend(
                _build_voxceleb_samples_for_roots(
                    args=args,
                    enabled_modalities=enabled_modalities,
                    speaker_to_label=speaker_to_label,
                    split_name=f"dev_{variant.name}",
                    video_root=base_video_root,
                    audio_root=variant / "audio_only",
                )
            )
    return samples


def _marine_species_from_path(path: Path):
    stem = path.stem.strip()
    if "_" not in stem:
        return stem

    # Drop the trailing sample index (e.g. *_15 or *_1209)
    base = stem.rsplit("_", 1)[0].strip()

    # Image files may include a source tag before the index, such as:
    # "<species>_matched_<n>" or "<species>_inat_<n>".
    # Remove that tag so species names align with audio filenames.
    for tag in ("_matched", "_inat"):
        if base.endswith(tag):
            base = base[: -len(tag)].strip()
            break

    return base


def _build_marine_samples_for_dirs(enabled_modalities, split_name: str, image_dir: Path, audio_dir: Path):
    images_by_species = defaultdict(list)
    audios_by_species = defaultdict(list)

    if "image" in enabled_modalities:
        if not image_dir.exists():
            print(f"[WARN] Marine image dir does not exist: {image_dir}", flush=True)
        else:
            for path in sorted(image_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    images_by_species[_marine_species_from_path(path)].append(path)

    if "audio" in enabled_modalities:
        if not audio_dir.exists():
            print(f"[WARN] Marine audio dir does not exist: {audio_dir}", flush=True)
        else:
            for path in sorted(audio_dir.iterdir()):
                if path.is_file() and path.suffix.lower() == ".wav":
                    audios_by_species[_marine_species_from_path(path)].append(path)

    samples = []
    if "image" in enabled_modalities and "audio" in enabled_modalities:
        for species in sorted(set(images_by_species) & set(audios_by_species)):
            for image_path in images_by_species[species]:
                for audio_path in audios_by_species[species]:
                    samples.append(
                        {
                            "dataset": "marine",
                            "split": split_name,
                            "sample_id": f"{species}__img={image_path.stem}__aud={audio_path.stem}",
                            "file": f"{image_path.name}|{audio_path.name}",
                            "text": "",
                            "audio": audio_path,
                            "video": None,
                            "image": image_path,
                            "label": species,
                        }
                    )
    elif "image" in enabled_modalities:
        for species, image_paths in sorted(images_by_species.items()):
            for image_path in image_paths:
                samples.append(
                    {
                        "dataset": "marine",
                        "split": split_name,
                        "sample_id": image_path.stem,
                        "file": image_path.name,
                        "text": "",
                        "audio": None,
                        "video": None,
                        "image": image_path,
                        "label": species,
                    }
                )
    else:
        for species, audio_paths in sorted(audios_by_species.items()):
            for audio_path in audio_paths:
                samples.append(
                    {
                        "dataset": "marine",
                        "split": split_name,
                        "sample_id": audio_path.stem,
                        "file": audio_path.name,
                        "text": "",
                        "audio": audio_path,
                        "video": None,
                        "image": None,
                        "label": species,
                    }
                )
    return samples


def _load_marine_samples(enabled_modalities, noisy_modalities, noise_severity: int | None = None):
    base_image_dir = Path("data/Marine/images")
    base_audio_dir = Path("data/Marine/audio")
    noise_root = Path("data/Marine/noise")

    if noisy_modalities is None:
        return _build_marine_samples_for_dirs(
            enabled_modalities=enabled_modalities,
            split_name="all",
            image_dir=base_image_dir,
            audio_dir=base_audio_dir,
        )

    samples = []
    if "image" in noisy_modalities:
        for variant in _iter_noise_variants(noise_root, "image", noise_severity):
            samples.extend(
                _build_marine_samples_for_dirs(
                    enabled_modalities=enabled_modalities,
                    split_name=f"all_{variant.name}",
                    image_dir=variant,
                    audio_dir=base_audio_dir,
                )
            )
    if "audio" in noisy_modalities:
        for variant in _iter_noise_variants(noise_root, "audio", noise_severity):
            samples.extend(
                _build_marine_samples_for_dirs(
                    enabled_modalities=enabled_modalities,
                    split_name=f"all_{variant.name}",
                    image_dir=base_image_dir,
                    audio_dir=variant,
                )
            )
    return samples
