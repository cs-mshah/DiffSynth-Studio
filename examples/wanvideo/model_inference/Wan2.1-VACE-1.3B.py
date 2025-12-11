import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)

# dataset_snapshot_download(
#     dataset_id="DiffSynth-Studio/examples_in_diffsynth",
#     local_dir="./",
#     allow_file_pattern=["data/examples/wan/depth_video.mp4", "data/examples/wan/cat_fightning.jpg"]
# )

# # Depth video -> Video
# control_video = VideoData("data/examples/wan/depth_video.mp4", height=480, width=832)
# video = pipe(
#     prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     vace_video=control_video,
#     seed=1, tiled=True
# )
# save_video(video, "video1.mp4", fps=15, quality=5)

# # Reference image -> Video
# video = pipe(
#     prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     vace_reference_image=Image.open("data/examples/wan/cat_fightning.jpg").resize((832, 480)),
#     seed=1, tiled=True
# )
# save_video(video, "video2.mp4", fps=15, quality=5)

# # Depth video + Reference image -> Video
# video = pipe(
#     prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     vace_video=control_video,
#     vace_reference_image=Image.open("data/examples/wan/cat_fightning.jpg").resize((832, 480)),
#     seed=1, tiled=True
# )
# save_video(video, "video3.mp4", fps=15, quality=5)

# gray video -> Video (inpainting)
pipe.load_lora(pipe.vace, "models/train/Wan2.1-VACE-1.3B_lora/epoch-3.safetensors", alpha=1)
pipe.enable_vram_management()

control_video = VideoData("data/example3/vid1_vace_video.mp4", height=480, width=832)
mask_video = VideoData("data/example3/vid1_vace_video_mask.mp4", height=480, width=832)
# control_video = VideoData("data/example_video_dataset/masked_output_gray.mp4", height=480, width=832)
# mask_video = VideoData("data/example_video_dataset/binary_mask_output.mp4", height=480, width=832)
video = pipe(
    prompt="""A full-sized fashion doll wearing a blue hoodie with a [V] white circular geometric logo centered on the chest,
            is standing upright in the middle of a brightly lit toyhouse play area. The background features colorful foam play 
            mats on the floor, miniature furniture, a slide, and various toys and pastel-colored storage shelves. The doll is 
            placed in the center foreground, facing the camera, with the camera moving softly in a front-facing arc around the 
            doll to keep the logo visible.""",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    # prompt="a photorealistic, cinematic, high-fashion commercial, a toy rubber duck floating on the swimming pool with the baby.",
    vace_video=control_video,
    vace_video_mask=mask_video,
    # vace_reference_image=Image.open("data/example3/ref_image.jpg").resize((832, 480)),
    seed=1, tiled=True
)
save_video(video, "data/example3/video_output1.mp4", fps=24, quality=9)
