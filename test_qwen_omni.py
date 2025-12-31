import soundfile as sf

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import torch
import os
import pandas as pd
import csv
import numpy as np
from parsing_util import get_label_for_file, get_utterance_text_for_file, extract_assistant_reply, get_ids_for_file

meta_data = pd.read_csv("data/MELD.Raw/test_sent_emo.csv")

# system prompt entry (reused for each conversation)
system_entry = {
    "role": "system",
    "content": [
        {"type": "text", "text": "The dataset contains utterances from Friends TV series. Each utterance in a dialog can be of positive, negative or neutral sentiment. Please classify the given sample by answering with exactly one word: neutral, negative or positive."},
    ],
}

# measure model load time
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
     "Qwen/Qwen2.5-Omni-7B",
     torch_dtype=torch.bfloat16,
     device_map="auto",
     attn_implementation="flash_attention_2",
)
model.disable_talker()

processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

TOTAL_SAMPLES = None
USE_AUDIO_IN_VIDEO = True

# collect mp4 files in deterministic order
files = sorted([f for f in os.listdir("data/MELD.Raw/output_repeated_splits_test") if f.endswith('.mp4')])
if TOTAL_SAMPLES is not None:
    files = files[:TOTAL_SAMPLES]

predictions = []

# path to save predictions and errors
out_path = os.path.join("out", "test_predictions.csv")
out_error_path = os.path.join("out", "test_error_prediction.csv")

device = next(model.parameters()).device
dtype = next(model.parameters()).dtype

for f in files:
    full_path = os.path.join("data/MELD.Raw/output_repeated_splits_test", f)
    utt_text = get_utterance_text_for_file(f, meta_data)

    # build single-sample conversation
    conversation = [system_entry, {"role": "user", "content": [
        {"type": "video", "video": full_path},
        {"type": "text", "text": utt_text},
    ]}]

    # Preparation for inference (single sample)
    try:
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = inputs.to(device).to(dtype)

        gen_output = model.generate(
            **inputs,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
            return_audio=False,
            output_scores=True,
            do_sample=False
        )
    except Exception as e:
        # append entry to error CSV
        dia_id, utt_id = get_ids_for_file(f)
        err_row = {"dialog_id": dia_id, "utterance_id": utt_id, "file": f, "error": str(e)}
        os.makedirs(os.path.dirname(out_error_path), exist_ok=True)
        write_header = not os.path.exists(out_error_path)
        with open(out_error_path, "a", newline='', encoding='utf-8') as errf:
            err_writer = csv.DictWriter(errf, fieldnames=["dialog_id", "utterance_id", "file", "error"])
            if write_header:
                err_writer.writeheader()
            err_writer.writerow(err_row)
        continue

    if hasattr(gen_output, "sequences"):
        text_ids = gen_output.sequences
    elif isinstance(gen_output, (list, tuple)):
        text_ids = gen_output[0]
    else:
        text_ids = gen_output

    # ensure tensor is on CPU / converted to numpy/list for the processor
    if isinstance(text_ids, torch.Tensor):
        text_ids = text_ids.cpu()

    decoded = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    out = decoded[0] if isinstance(decoded, (list, tuple)) else decoded
    reply = extract_assistant_reply(out)
    dia_id, utt_id = get_ids_for_file(f)
    label = get_label_for_file(f, meta_data)
    new_row = {"dialog_id": dia_id, "utterance_id": utt_id, "file": f, "prediction": reply, "label": label}
    predictions.append(new_row)

    # Append only the new prediction to CSV (create header if file doesn't exist)
    write_header = not os.path.exists(out_path)
    with open(out_path, "a", newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["dialog_id", "utterance_id", "file", "prediction", "label"]) 
        if write_header:
            writer.writeheader()
        writer.writerow(new_row)

