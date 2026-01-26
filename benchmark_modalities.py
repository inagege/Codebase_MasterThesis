import sys
import time
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import os
import csv
from pathlib import Path

import pandas as pd
import torch

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

from utils.parsing_util import (
    get_label_for_file,
    get_utterance_text_for_file,
    extract_assistant_reply,
    get_ids_for_file,
)

# -------------------------
# CLI: modality parameters
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--modalities",
        type=str,
        default="text,video,audio",
        help="Comma-separated list: any of text,audio,video. Example: text or text,audio or text,video",
    )
    p.add_argument(
        "--noisy-modalities",
        type=str,
        default=None,
        help="State which of the input modalities should use noisy input."
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        help="The split to process (train, val, test)."
    )
    p.add_argument(
        "--use-audio-in-video",
        action="store_true",
        help="(Ignored) Audio is expected as WAV under audio_only/. Kept for compatibility.",
    )    
    p.add_argument("--total-samples", type=int, default=None, help="Limit total files across all splits")
    p.add_argument("--audio-subdir", type=str, default="audio_only", help="Subdir under each split dir with WAVs")
    p.add_argument("--out-path", type=str, default="out/prediction_noise.csv")
    p.add_argument("--out-error-path", type=str, default="out/error_prediction_noise.csv")
    return p.parse_args()


def normalize_modalities(mod_str: str) -> set[str]:
    if mod_str is None:
        return mod_str
    mods = {m.strip().lower() for m in mod_str.split(",") if m.strip()}
    valid = {"text", "audio", "video"}
    bad = mods - valid
    if bad:
        raise ValueError(f"Unknown modalities: {bad}. Valid: {sorted(valid)}")
    if not mods:
        raise ValueError("No modalities selected. Use --modalities text,audio,video (any subset).")
    return mods

def normalize_splits(split_str: str) -> set[str]:
    splits = {s.strip().lower() for s in split_str.split(",") if s.strip()}
    valid = {"test", "train", "val"}
    bad = splits - valid
    if bad:
        raise ValueError(f"Unknown modalities: {bad}. Valid: {sorted(valid)}")
    if not splits:
        raise ValueError("No split selected. Use --split test,train,val (any subset).")
    return splits

def get_root(split: str) -> str:
    match split:
        case 'train':
            return 'data/MELD.Raw/train_splits'
        case 'test':
            return 'data/MELD.Raw/output_repeated_splits_test'
        case 'val':
            return 'data/MELD.Raw/dev_splits_complete'
        case _:
            raise ValueError("No split selected. Use --split test,train,val (any subset).")
        
def get_meta_csv(split: str) -> str:
    match split:
        case 'train':
            return 'train_sent_emo.csv'
        case 'test':
            return 'test_sent_emo.csv'
        case 'val':
            return 'dev_sent_emo.csv'
        case _:
            raise ValueError("No split selected. Use --split test,train,val (any subset).")
		    

def get_split_configs(noisy_modalities: set[str], split: str) -> list[tuple[str, str, str]]:
    SPLIT_CONFIGS = []
    root = get_root(split=split)

    if noisy_modalities is None:
        path = (os.path.join(root,'unmodified'))
        meta_data = os.path.join(root, get_meta_csv(split))
        ref_split = 'unmodified'
        SPLIT_CONFIGS.append((path, meta_data, ref_split))
    else:
        meta = get_meta_csv(split=split)

        for mod in noisy_modalities - {'text'}:
            path = os.path.join(root, mod)
            for name in os.listdir(path):
                if os.path.isdir(os.path.join(path, name)):
                    SPLIT_CONFIGS.append((os.path.join(path, name), meta, name))
        if 'text' in noisy_modalities:
            path = os.path.join(root, mod)
            for name in os.listdir(path):
                if os.path.isdir(os.path.join(path, name)):
                    SPLIT_CONFIGS.append((os.path.join(root,'‘unmodified'), os.path.join(path, name, 'metadata.csv'), name))
                
    return SPLIT_CONFIGS




def audio_path_for_mp4(mp4_path: Path, audio_subdir: str) -> Path:
    # mp4_path is .../<split_dir>/<file>.mp4
    # audio is  .../<split_dir>/<audio_subdir>/<file>.wav
    return mp4_path.parent / audio_subdir / (mp4_path.stem + ".wav")


def main():
    args = parse_args()
    enabled = normalize_modalities(args.modalities)
    noisy = normalize_modalities(args.noisy_modalities)
    SPLIT_CONFIGS = get_split_configs(noisy_modalities=noisy, split=args.split)
    print(SPLIT_CONFIGS)

    system_entry = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "The dataset contains utterances from Friends TV series. "
                    "Each utterance in a dialog can be of positive, negative or neutral sentiment. "
                    "Please classify the given sample by answering with exactly one word: "
                    "neutral, negative or positive."
                ),
            },
        ],
    }

    print(f"[INFO] CWD: {os.getcwd()}", flush=True)
    print(f"[INFO] Modalities enabled: {sorted(enabled)}", flush=True)
    print(f"[INFO] Noisy modalities: {args.noisy_modalities}", flush=True)
    print(f"[INFO] Split: {args.split}", flush=True)
    print(f"[INFO] audio_subdir={args.audio_subdir}", flush=True)
    print(f"[INFO] out_path={args.out_path}", flush=True)
    print(f"[INFO] out_error_path={args.out_error_path}", flush=True)

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_error_path) or ".", exist_ok=True)
    os.makedirs("out", exist_ok=True)

    # Load metadata
    meta_map = {}
    for _dir, meta_csv, split in SPLIT_CONFIGS:
        try:
            meta_map[split] = pd.read_csv(meta_csv)
            print(f"[INFO] Loaded meta: {meta_csv} rows={len(meta_map[split])}", flush=True)
        except Exception as e:
            meta_map[split] = None
            print(f"[WARN] Could not load meta: {meta_csv} error={e}", flush=True)

    # Collect MP4s (top-level only; change to rglob if needed)
    files = []
    for _dir, _meta_csv, split in SPLIT_CONFIGS:
        d = Path(_dir)
        print(f"[INFO] Scanning dir: {_dir} (exists={d.exists()})", flush=True)
        if not d.exists():
            continue

        mp4s = sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"])
        print(f"[INFO] Found {len(mp4s)} mp4 files in {_dir}", flush=True)

        for p in mp4s:
            files.append((p.name, p, split))  # keep Path object

    if args.total_samples is not None:
        files = files[: args.total_samples]

    print(f"[INFO] Total files to process: {len(files)}", flush=True)
    if len(files) == 0:
        raise RuntimeError("No MP4 files found (top-level). If nested, use rglob('*.mp4').")

    # Load model
    print("[INFO] Loading model...", flush=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    print("[INFO] Model loaded.", flush=True)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    print(f"[INFO] device={device} dtype={dtype}", flush=True)

    fieldnames = ["dialog_id", "utterance_id", "file", "prediction", "label", "split"]
    err_fieldnames = ["dialog_id", "utterance_id", "file", "error", "traceback"]

    for i, (fname, mp4_path, split) in enumerate(files, start=1):
        meta_data = meta_map.get(split)
        utt_text = get_utterance_text_for_file(fname, meta_data)

        dia_id, utt_id = get_ids_for_file(fname)
        label = get_label_for_file(fname, meta_data)

        # Build user content based on enabled modalities
        user_content = []
        if "video" in enabled:
            user_content.append({"type": "video", "video": str(mp4_path)})

        if "audio" in enabled:
            wav_path = audio_path_for_mp4(mp4_path, args.audio_subdir)
            if not wav_path.exists():
                # Log missing audio and continue
                err_row = {
                    "dialog_id": dia_id,
                    "utterance_id": utt_id,
                    "file": fname,
                    "error": f"Missing audio wav: {wav_path}",
                    "traceback": "",
                }
                write_header = not os.path.exists(args.out_error_path)
                with open(args.out_error_path, "a", newline="", encoding="utf-8") as errf:
                    err_writer = csv.DictWriter(errf, fieldnames=err_fieldnames)
                    if write_header:
                        err_writer.writeheader()
                    err_writer.writerow(err_row)
                continue

            # IMPORTANT: add AUDIO to the conversation (not video)
            user_content.append({"type": "audio", "audio": str(wav_path)})

        if "text" in enabled:
            user_content.append({"type": "text", "text": utt_text})

        conversation = [system_entry, {"role": "user", "content": user_content}]

        try:
            text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

            # Since audio is pre-extracted wav, do NOT use audio-in-video
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

            proc_kwargs = dict(
                text=text_prompt,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False,
                images=None,
            )

            if "audio" in enabled:
                proc_kwargs["audio"] = audios
            if "video" in enabled:
                proc_kwargs["videos"] = videos
            else:
                proc_kwargs["videos"] = None

            inputs = processor(**proc_kwargs).to(device).to(dtype)

            gen_output = model.generate(
                **inputs,
                use_audio_in_video=False,
                return_audio=False,
                output_scores=True,
                do_sample=False,
            )

            text_ids = (
                gen_output.sequences
                if hasattr(gen_output, "sequences")
                else gen_output[0]
                if isinstance(gen_output, (list, tuple))
                else gen_output
            )
            if isinstance(text_ids, torch.Tensor):
                text_ids = text_ids.cpu()

            decoded = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            out = decoded[0] if isinstance(decoded, (list, tuple)) else decoded
            reply = extract_assistant_reply(out)

            new_row = {
                "dialog_id": dia_id,
                "utterance_id": utt_id,
                "file": fname,
                "prediction": reply,
                "label": label,
                "split": split,
            }

            write_header = not os.path.exists(args.out_path)
            with open(args.out_path, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(new_row)

            if i % 20 == 0 or i == 1:
                print(f"[INFO] Wrote prediction {i}/{len(files)} -> {args.out_path}", flush=True)

        except Exception as e:
            err_row = {
                "dialog_id": dia_id,
                "utterance_id": utt_id,
                "file": fname,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            write_header = not os.path.exists(args.out_error_path)
            with open(args.out_error_path, "a", newline="", encoding="utf-8") as errf:
                err_writer = csv.DictWriter(errf, fieldnames=err_fieldnames)
                if write_header:
                    err_writer.writeheader()
                err_writer.writerow(err_row)


if __name__ == "__main__":
    main()
