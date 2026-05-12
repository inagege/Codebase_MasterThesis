import re
from pathlib import Path

PREDICTION_FILE_RE = re.compile(r"^prediction_(?P<modalities>[a-z]+)_noise_(?P<noise>[a-z]*)\.csv$", re.IGNORECASE)
KV_PATTERN = re.compile(r"([A-Za-z]+)=([^=]+?)(?=_[A-Za-z]+=|$)")


def parse_modalities_from_prediction_filename(path: Path) -> dict[str, str]:
    match = PREDICTION_FILE_RE.match(path.name)
    if not match:
        return {"modalities": "", "noise_modalities": ""}
    return {
        "modalities": (match.group("modalities") or "").lower(),
        "noise_modalities": (match.group("noise") or "").lower(),
    }


def parse_split_metadata(split: str) -> dict[str, object]:
    split_text = str(split or "").strip()
    split_lower = split_text.lower()

    if split_lower in {"all", "test_all", "dev"} or "unmodified" in split_lower:
        return {
            "split": split_text,
            "is_unmodified": True,
            "severity": None,
            "perturbation_method": "unmodified",
            "perturbation_target": "",
        }

    pairs = {key.upper(): value for key, value in KV_PATTERN.findall(split_text)}
    severity_raw = pairs.get("S")
    severity = None
    if severity_raw is not None:
        try:
            severity = int(severity_raw)
        except ValueError:
            severity = None

    method_values = []
    method_targets = []
    for key, value in pairs.items():
        if key == "S":
            continue
        method_targets.append(key.lower())
        method_values.append(value.lower())

    perturbation_method = "+".join(method_values) if method_values else "unknown"
    perturbation_target = "+".join(method_targets)

    return {
        "split": split_text,
        "is_unmodified": False,
        "severity": severity,
        "perturbation_method": perturbation_method,
        "perturbation_target": perturbation_target,
    }
