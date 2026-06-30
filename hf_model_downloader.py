import argparse
import os
import yaml
from huggingface_hub import snapshot_download

model_repo = "google/gemma-4-12B"
save_dir = "/home/ai-model/LLM/model"

################################################################################################################
################################################################################################################
################################################################################################################

print(f"Downloading '{model_repo}' -> '{save_dir}'")
os.makedirs(save_dir, exist_ok=True)

local_path = snapshot_download(
    repo_id=model_repo,
    local_dir=save_dir,
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
)

print(f"Download complete: {local_path}")
