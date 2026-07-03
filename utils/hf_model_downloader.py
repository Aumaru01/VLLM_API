import argparse
import os
import yaml
from huggingface_hub import snapshot_download
from pathlib import Path
import re

_cfg_path = Path(__file__).parent / "model_download_config.yaml"
with open(_cfg_path) as f:
    _cfg = yaml.safe_load(f)

model_repo = _cfg["model_name"]
save_dir = _cfg["save_dir"]
hf_token = _cfg["Huggingface_token"]

################################################################################################################
################################################################################################################
################################################################################################################

save_dir = Path(save_dir,Path(re.sub(r"\.", "_", model_repo)).stem)

print(f"Downloading '{model_repo}' -> '{save_dir}'")
os.makedirs(save_dir, exist_ok=True)

local_path = snapshot_download(
    repo_id=model_repo,
    local_dir=save_dir,
    token=hf_token,
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
)

print(f"Download complete: {local_path}")
