DATA_FOLDER="data/example3/"

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path "${DATA_FOLDER}" \
  --dataset_metadata_path "${DATA_FOLDER}/metadata_no_prompt.csv" \
  --data_file_keys "video,vace_video,vace_video_mask" \
  --height 480 \
  --width 832 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth" \
  --remove_prefix_in_ckpt "pipe.vace." \
  --lora_base_model "vace" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "vace_video,vace_video_mask" \
  --use_gradient_checkpointing_offload \
  --task data_process  \
  --output_path "${DATA_FOLDER}/full_caption_cache"