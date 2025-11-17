import torch
from PIL import Image
from torch.serialization import add_safe_globals
import argparse, pprint

add_safe_globals([Image.Image])

def summarize(path):
    data = torch.load(path, map_location='cpu', weights_only=False)
    print(f"Loaded {path}")
    print(f"Type: {type(data)} | Keys: {list(data.keys())}")
    for key, value in data.items():
        if isinstance(value, list):
            info = f"list[{len(value)}]"
            if len(value) > 0:
                info += f" of {type(value[0])}"
            print(f"  {key}: {info}")
        else:
            print(f"  {key}: {type(value)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()
    summarize(args.path)

# Example usage:
# python inspect_pth.py data/example_video_dataset/Wan2.1-VACE-1.3B_lora/0/0.pth