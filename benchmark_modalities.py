import argparse
import csv
import os
import traceback
from pathlib import Path

import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from utils.benchmark_data_loading import (
    default_modalities_for_dataset,
    get_prompt_for_classification,
    load_samples,
    normalize_dataset_name,
    normalize_meld_task,
    normalize_modalities,
    validate_modalities,
)
from utils.parsing_util import extract_assistant_reply


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="meld",
        help="Dataset to benchmark: meld, homeprice, imdb, voxceleb, nejm, marine.",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default=None,
        help="Comma-separated list from text,audio,video,image. If omitted, all modalities available in the selected dataset are used.",
    )
    parser.add_argument(
        "--noisy-modalities",
        type=str,
        default=None,
        help="Comma-separated modalities that should use noisy input variants.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split(s) to process. For MELD: train,val,test (comma-separated). Others: ignored and treated as all/dev.",
    )
    parser.add_argument(
        "--classification-task",
        type=str,
        default=None,
        help="MELD only: sentiment|emotion.",
    )
    parser.add_argument(
        "--start-at-sample",
        type=int,
        default=None,
        help="Optional lower bound (0-based). If set, skip all samples before this index.",
    )
    parser.add_argument("--total-samples", type=int, default=None, help="Limit total files across all splits")
    parser.add_argument("--audio-subdir", type=str, default="audio_only", help="Subdir for WAV files")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of samples per generation batch. Reduce if you hit CUDA OOM.",
    )
    parser.add_argument("--out-path", type=str, default="out/prediction_noise.csv")
    parser.add_argument("--out-error-path", type=str, default="out/error_prediction_noise.csv")
    return parser.parse_args()


def append_csv_row(path: str, fieldnames: list[str], row: dict):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def cast_floats_to_dtype(batch, dtype: torch.dtype):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
            batch[key] = value.to(dtype)
    return batch


def log_error_row(args, err_fieldnames, dataset, sample, error_text, traceback_text=""):
    append_csv_row(
        args.out_error_path,
        err_fieldnames,
        {
            "dataset": dataset,
            "split": sample["split"],
            "sample_id": sample["sample_id"],
            "file": sample["file"],
            "error": error_text,
            "traceback": traceback_text,
        },
    )


def build_user_content(enabled_modalities, sample, args, err_fieldnames, dataset):
    user_content = []
    skip_sample = False

    if "video" in enabled_modalities:
        video_path = sample.get("video")
        if video_path is None or not Path(video_path).exists():
            log_error_row(args, err_fieldnames, dataset, sample, f"Missing video file: {video_path}")
            skip_sample = True
        else:
            user_content.append({"type": "video", "video": str(video_path)})

    if "audio" in enabled_modalities:
        audio_path = sample.get("audio")
        if audio_path is None or not Path(audio_path).exists():
            log_error_row(args, err_fieldnames, dataset, sample, f"Missing audio file: {audio_path}")
            skip_sample = True
        else:
            user_content.append({"type": "audio", "audio": str(audio_path)})

    if "image" in enabled_modalities:
        image_path = sample.get("image")
        if image_path is None or not Path(image_path).exists():
            log_error_row(args, err_fieldnames, dataset, sample, f"Missing image file: {image_path}")
            skip_sample = True
        else:
            user_content.append({"type": "image", "image": str(image_path)})

    if "text" in enabled_modalities:
        text_value = sample.get("text", "").strip()
        if not text_value:
            log_error_row(args, err_fieldnames, dataset, sample, "Missing text input.")
            skip_sample = True
        else:
            user_content.append({"type": "text", "text": text_value})

    return user_content, skip_sample


def build_conversation_for_sample(sample, dataset, prompt, enabled_modalities, args, err_fieldnames):
    sample_prompt = prompt
    if dataset == "nejm":
        options = (sample.get("options") or "").strip()
        if options:
            sample_prompt = f"{prompt} Options: {options}"

    system_entry = {
        "role": "system",
        "content": [{"type": "text", "text": sample_prompt}],
    }
    print(sample_prompt)

    user_content, skip_sample = build_user_content(
        enabled_modalities=enabled_modalities,
        sample=sample,
        args=args,
        err_fieldnames=err_fieldnames,
        dataset=dataset,
    )
    if skip_sample:
        return None

    return [system_entry, {"role": "user", "content": user_content}]


def run_batch_generation(model, processor, entries, enabled_modalities, device, dtype):
    conversations = [entry["conversation"] for entry in entries]
    text_prompt = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)

    proc_kwargs = {
        "text": text_prompt,
        "return_tensors": "pt",
        "padding": True,
        "use_audio_in_video": False,
    }
    if "audio" in enabled_modalities:
        proc_kwargs["audio"] = audios
    if "video" in enabled_modalities:
        proc_kwargs["videos"] = videos
    if "image" in enabled_modalities:
        proc_kwargs["images"] = images

    inputs = processor(**proc_kwargs).to(device)
    inputs = cast_floats_to_dtype(inputs, dtype)

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
    if isinstance(decoded, str):
        decoded = [decoded]
    if len(decoded) != len(entries):
        raise RuntimeError(
            f"Decoded outputs ({len(decoded)}) do not match batch size ({len(entries)})."
        )

    return [extract_assistant_reply(out_text) for out_text in decoded]


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.start_at_sample is not None and args.start_at_sample < 0:
        raise ValueError("--start-at-sample must be >= 0 when provided.")

    dataset = normalize_dataset_name(args.dataset)
    enabled_modalities = normalize_modalities(args.modalities)
    if enabled_modalities is None:
        enabled_modalities = default_modalities_for_dataset(dataset)

    noisy_modalities = normalize_modalities(args.noisy_modalities)
    validate_modalities(dataset, enabled_modalities, noisy_modalities)

    meld_task = None
    label_column = None
    if dataset == "meld":
        meld_task, label_column = normalize_meld_task(args.classification_task)
    elif dataset == "voxceleb":
        label_column = "nationality_wiki"

    prompt = get_prompt_for_classification(dataset, meld_task)

    print(f"[INFO] CWD: {os.getcwd()}", flush=True)
    print(f"[INFO] Dataset: {dataset}", flush=True)
    print(f"[INFO] Modalities enabled: {sorted(enabled_modalities)}", flush=True)
    print(f"[INFO] Noisy modalities: {args.noisy_modalities}", flush=True)
    print(f"[INFO] Split: {args.split}", flush=True)
    print(f"[INFO] audio_subdir={args.audio_subdir}", flush=True)
    print(f"[INFO] batch_size={args.batch_size}", flush=True)
    print(f"[INFO] start_at_sample={args.start_at_sample}", flush=True)
    if label_column is not None:
        print(f"[INFO] Label column: {label_column}", flush=True)
    print(f"[INFO] out_path={args.out_path}", flush=True)
    print(f"[INFO] out_error_path={args.out_error_path}", flush=True)

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_error_path) or ".", exist_ok=True)
    os.makedirs("out", exist_ok=True)

    samples = load_samples(dataset, args, enabled_modalities, noisy_modalities, label_column)
    if args.start_at_sample is not None:
        samples = samples[args.start_at_sample :]
    if args.total_samples is not None:
        samples = samples[: args.total_samples]

    print(f"[INFO] Total samples to process: {len(samples)}", flush=True)
    if not samples:
        raise RuntimeError("No samples found for the selected dataset/modality configuration.")

    print("[INFO] Loading model...", flush=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        enable_audio_output=False,
    )
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    print("[INFO] Model loaded.", flush=True)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    print(f"[INFO] device={device} dtype={dtype}", flush=True)

    fieldnames = [
        "dataset",
        "split",
        "sample_id",
        "file",
        "prediction",
        "label",
    ]
    err_fieldnames = [
        "dataset",
        "split",
        "sample_id",
        "file",
        "error",
        "traceback",
    ]

    num_written = 0
    for batch_start in range(0, len(samples), args.batch_size):
        batch_samples = samples[batch_start : batch_start + args.batch_size]
        entries = []

        for sample in batch_samples:
            conversation = build_conversation_for_sample(
                sample=sample,
                dataset=dataset,
                prompt=prompt,
                enabled_modalities=enabled_modalities,
                args=args,
                err_fieldnames=err_fieldnames,
            )
            if conversation is None:
                continue
            entries.append({"sample": sample, "conversation": conversation})

        if not entries:
            continue

        try:
            replies = run_batch_generation(
                model=model,
                processor=processor,
                entries=entries,
                enabled_modalities=enabled_modalities,
                device=device,
                dtype=dtype,
            )
        except Exception:
            print(
                f"[WARN] Batch {batch_start + 1}-{batch_start + len(batch_samples)} failed; retrying individually.",
                flush=True,
            )
            for entry in entries:
                sample = entry["sample"]
                try:
                    replies = run_batch_generation(
                        model=model,
                        processor=processor,
                        entries=[entry],
                        enabled_modalities=enabled_modalities,
                        device=device,
                        dtype=dtype,
                    )
                    reply = replies[0]
                    append_csv_row(
                        args.out_path,
                        fieldnames,
                        {
                            "dataset": dataset,
                            "split": sample["split"],
                            "sample_id": sample["sample_id"],
                            "file": sample["file"],
                            "prediction": reply,
                            "label": sample.get("label", "unknown"),
                        },
                    )
                    num_written += 1
                    if num_written % 20 == 0 or num_written == 1:
                        print(f"[INFO] Wrote prediction {num_written}/{len(samples)} -> {args.out_path}", flush=True)
                except Exception as exc:
                    log_error_row(
                        args=args,
                        err_fieldnames=err_fieldnames,
                        dataset=dataset,
                        sample=sample,
                        error_text=str(exc),
                        traceback_text=traceback.format_exc(),
                    )
            continue

        for entry, reply in zip(entries, replies):
            sample = entry["sample"]
            append_csv_row(
                args.out_path,
                fieldnames,
                {
                    "dataset": dataset,
                    "split": sample["split"],
                    "sample_id": sample["sample_id"],
                    "file": sample["file"],
                    "prediction": reply,
                    "label": sample.get("label", "unknown"),
                },
            )
            num_written += 1
            if num_written % 20 == 0 or num_written == 1:
                print(f"[INFO] Wrote prediction {num_written}/{len(samples)} -> {args.out_path}", flush=True)


if __name__ == "__main__":
    main()
