#!/usr/bin/env python3
import argparse
import random
import string
from pathlib import Path

import pandas as pd

TEXT_CORRUPTIONS = ["keyboard", "char_replace", "ocr", "char_delete", "top4_paper"]

QWERTY_NEIGHBORS = {
    "a": "qwsxz", "b": "vghn", "c": "xsdfv", "d": "serfcx", "e": "wsdr",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu", "z": "asx",
}

OCR_CONFUSIONS = {"o": "0", "O": "0", "i": "1", "I": "1", "l": "1", "e": "3", "a": "@", "s": "5", "S": "5", "b": "6", "g": "9"}

def severity_to_rate(sev: int) -> float:
    sev = max(1, min(5, sev))
    return {1: 0.03, 2: 0.06, 3: 0.10, 4: 0.15, 5: 0.20}[sev]

def _pick_indices(chars, predicate, k, rng):
    idxs = [i for i, ch in enumerate(chars) if predicate(ch)]
    if not idxs:
        return []
    return rng.sample(idxs, k=min(k, len(idxs)))

def keyboard(text: str, rate: float, rng: random.Random) -> str:
    chars = list(text)
    k = max(1, int(len(chars) * rate))
    chosen = _pick_indices(chars, lambda ch: ch.lower() in QWERTY_NEIGHBORS, k, rng)
    for i in chosen:
        ch = chars[i]
        repl = rng.choice(QWERTY_NEIGHBORS[ch.lower()])
        chars[i] = repl.upper() if ch.isupper() else repl
    return "".join(chars)

def char_replace(text: str, rate: float, rng: random.Random) -> str:
    chars = list(text)
    k = max(1, int(len(chars) * rate))
    chosen = _pick_indices(chars, lambda ch: ch.isalpha(), k, rng)
    for i in chosen:
        ch = chars[i]
        repl = rng.choice(string.ascii_lowercase)
        chars[i] = repl.upper() if ch.isupper() else repl
    return "".join(chars)

def ocr(text: str, rate: float, rng: random.Random) -> str:
    chars = list(text)
    k = max(1, int(len(chars) * rate))
    chosen = _pick_indices(chars, lambda ch: ch in OCR_CONFUSIONS, k, rng)
    for i in chosen:
        chars[i] = OCR_CONFUSIONS[chars[i]]
    return "".join(chars)

def char_delete(text: str, rate: float, rng: random.Random) -> str:
    chars = list(text)
    k = max(1, int(len(chars) * rate))
    chosen = sorted(_pick_indices(chars, lambda ch: not ch.isspace(), k, rng), reverse=True)
    for i in chosen:
        chars.pop(i)
    return "".join(chars)

def perturb(text: str, method: str, severity: int, rng: random.Random) -> str:
    if not isinstance(text, str):
        return text
    rate = severity_to_rate(severity)
    if method == "keyboard":
        return keyboard(text, rate, rng)
    if method == "char_replace":
        return char_replace(text, rate, rng)
    if method == "ocr":
        return ocr(text, rate, rng)
    if method == "char_delete":
        return char_delete(text, rate, rng)
    if method == "top4_paper":
        r = rate * 0.25
        t = keyboard(text, r, rng)
        t = char_replace(t, r, rng)
        t = ocr(t, r, rng)
        t = char_delete(t, r, rng)
        return t
    raise ValueError(method)

def main():
    ap = argparse.ArgumentParser("Apply ALL text perturbations to a CSV 'Utterance' column.")
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_dir", required=True, help="Base output directory.")
    ap.add_argument("--severity", type=int, default=3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--output_column", default="Utterance_noisy")
    args = ap.parse_args()

    inp = Path(args.input_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    if "Utterance" not in df.columns:
        raise ValueError("CSV must contain a column named 'Utterance'")

    for corr in TEXT_CORRUPTIONS:
        combo_root = out_dir / f"T={corr}_S={args.severity}"
        combo_root.mkdir(parents=True, exist_ok=True)

        rng = random.Random(args.seed)  # reset per corruption for reproducibility
        df_out = df.copy()
        df_out[args.output_column] = df_out["Utterance"].apply(lambda t: perturb(t, corr, args.severity, rng))

        out_csv = combo_root / "metadata.csv"
        df_out.to_csv(out_csv, index=False)
        print(f"[OK] Wrote {out_csv}")

    print("[DONE] All text perturbations applied.")

if __name__ == "__main__":
    main()
