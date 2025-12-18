import soundfile as sf

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import torch
from datetime import datetime
import pandas as pd
import time


train_data = pd.read_csv("data/MELD.Raw/train_sent_emo.csv")
sample = "dia0_utt0.mp4"
utterance_id = sample[sample.find("_utt")+4:sample.find(".mp4", sample.find("_utt"))]
utterance = train_data[train_data["Sr No."] == utterance_id]["Utterance"]

print("Loading Qwen2.5-Omni-7B model at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
# measure model load time
t_load_start = time.time()
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
     "Qwen/Qwen2.5-Omni-7B",
     torch_dtype=torch.bfloat16,
     device_map="auto",
     attn_implementation="flash_attention_2",
 )
t_load_end = time.time()
print(f"Model loaded in {t_load_end - t_load_start:.2f}s")

processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
print("Processor loaded at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("torch.cuda.is_available():", torch.cuda.is_available())
print("model device:", next(model.parameters()).device)
print("model dtype:", next(model.parameters()).dtype)

# Warm-up short generation to trigger kernel compilation / allocations
try:
    warm_conv = [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    ]
    warm_text = processor.apply_chat_template(warm_conv, add_generation_prompt=True, tokenize=False)
    warm_inputs = processor(text=warm_text, return_tensors="pt", padding=True)
    warm_device = next(model.parameters()).device
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    warm_inputs = warm_inputs.to(warm_device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_w0 = time.time()
    # small generation for warm-up
    _ = model.generate(**warm_inputs, max_new_tokens=8)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_w1 = time.time()
    print(f"Warm-up generation took {t_w1 - t_w0:.2f}s")
except Exception as e:
    print("Warm-up failed:", e)

conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "The dataset contains utterances from Friends TV series. Each utterance in a dialog can be of positive, negative or neutral sentiment. Please classify the given sample as: neutral, negative or positive."},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "data/MELD.Raw/train_splits/dia0_utt0.mp4"},
            {"type": "text", "text": utterance}
        ],
    },
]

# set use audio in video
USE_AUDIO_IN_VIDEO = True

# Preparation for inference
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)

# measure device transfer time for inputs
device = next(model.parameters()).device
dtype = next(model.parameters()).dtype
if torch.cuda.is_available():
    torch.cuda.synchronize()
t_transfer_start = time.time()
inputs = inputs.to(device).to(dtype)
if torch.cuda.is_available():
    torch.cuda.synchronize()
t_transfer_end = time.time()
print(f"Inputs transfer to device took {t_transfer_end - t_transfer_start:.2f}s")

# Inference: Generation of the output text and audio
if torch.cuda.is_available():
    torch.cuda.synchronize()
t_gen_start = time.time()
text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, temperature=0, top_p=1)
if torch.cuda.is_available():
    torch.cuda.synchronize()
t_gen_end = time.time()
print(f"Generation took {t_gen_end - t_gen_start:.2f}s")

text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(text)

