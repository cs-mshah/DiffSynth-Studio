import os
import torch
import numpy as np
import pandas as pd
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from tqdm import tqdm

np.random.seed(42)
output_dir = "data/example3/outputs/5epochs_32_rank"
input_dir = "data/example3/preprocess_target/"
df = pd.read_csv("data/example3/preprocess_target/metadata_target.csv")
os.makedirs(output_dir, exist_ok=True)
num_inference = 1
lora_path = "models/train/Wan2.1-VACE-1.3B_lora/example3/5epochs/epoch-4.safetensors"

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)
# correctly load lora
pipe.load_lora(pipe.vace, lora_path, alpha=1)
pipe.enable_vram_management()

print(f'starting inference')
for index, row in tqdm(df.iterrows()):
    print(f'inferrring {row["video"]}...')
    control_video_path = os.path.join(input_dir, row["vace_video"])
    control_video = VideoData(control_video_path, height=480, width=832)
    mask_video_path = os.path.join(input_dir, row["vace_video_mask"])
    mask_video = VideoData(mask_video_path, height=480, width=832)
    prompt = row["prompt"]

    for i in range(num_inference):
        seed = np.random.randint(1, 10000001)
        video = pipe(
            prompt=prompt,
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            vace_video=control_video,
            vace_video_mask=mask_video,
            seed=seed, tiled=True
        )
        save_video(video, f"{output_dir}/{row['video'].split('.')[0]}_seed_{seed}.mp4", fps=24, quality=9)
