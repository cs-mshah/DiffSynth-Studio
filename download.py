from huggingface_hub import snapshot_download
import os

local_dir = "./target3"
snapshot_download(
    repo_id="VLR16824/vlr_data",
    repo_type="dataset",
    local_dir=local_dir,
    token=os.environ["HF_TOKEN"],
)

# from huggingface_hub import snapshot_download, HfApi
# import os
# os.environ["HF_TOKEN"] = ""

# api = HfApi()
# repo_id = "VLR16824/vlr_data"

# snapshot_download(repo_id=repo_id, local_dir="data/raw", token=os.environ["HF_TOKEN"], repo_type="dataset")

# # allowed_patterns = ["*.pth"]
# api.upload_folder(
#     folder_path="data/example3/preprocess",
#     repo_id=repo_id,
#     path_in_repo="example3/preprocess",
#     repo_type="dataset",
#     token=os.environ["HF_TOKEN"]
# )


# api.hf_hub_download(repo_id,
#                 filename="example3/tmp/epoch-3.safetensors",
#                 repo_type="dataset",
#                 local_dir="models/train/Wan2.1-VACE-1.3B_lora/example3/ricky")