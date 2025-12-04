import os
import requests
import torch
import soundfile as sf

# First: enable MPS fallback to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# Download video only once
video_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"
video_path = "draw.mp4"
if not os.path.exists(video_path):
    resp = requests.get(video_url)
    resp.raise_for_status()
    with open(video_path, "wb") as f:
        f.write(resp.content)

# automatically set device and dtype
if torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = torch.float16
elif torch.backends.mps.is_available():
    device = torch.device("cpu") 
    dtype = torch.float16

# Load model + tokenizer / processor
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-3B",
    torch_dtype=dtype,
)
model = model.to(device)

processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")

conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team …"}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": video_path},
        ],
    },
]

USE_AUDIO_IN_VIDEO = True

# Prepare inputs
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(
    text=text, audio=audios, images=images, videos=videos,
    return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO
)

# Move inputs to device
inputs = {k: v.to(device) for k, v in inputs.items()}

# Run inference
text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)

text = processor.batch_decode(
    text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(text)

sf.write(
    "output.wav",
    audio.reshape(-1).detach().cpu().numpy(),
    samplerate=24000,
)
