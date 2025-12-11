DATA_FOLDER="data/example6/preprocess/ohwx_caption_cache/0"
DATASET_REPEAT=16
LORA_RANK=32
MASKED_LOSS_LAMBDA=0.9
MODEL_OUTPUT_PATH="./models/train/Wan2.1-VACE-1.3B_lora/example6/ohwx_caption"

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path "${DATA_FOLDER}" \
  --data_file_keys "video,vace_video,vace_video_mask" \
  --height 480 \
  --width 832 \
  --dataset_repeat $DATASET_REPEAT \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "${MODEL_OUTPUT_PATH}" \
  --lora_base_model "vace" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank $LORA_RANK \
  --extra_inputs "vace_video,vace_video_mask" \
  --use_gradient_checkpointing_offload \
  --masked_mse_lambda $MASKED_LOSS_LAMBDA