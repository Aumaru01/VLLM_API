#0 if use environment(venv)
source /venv/bin/activate

# Fix NVRTC builtins lookup for the cu13 torch wheel (LD_LIBRARY_PATH otherwise
# resolves to /usr/local/cuda -> CUDA 12.2, which lacks libnvrtc-builtins.so.13.0)
export LD_LIBRARY_PATH="$(pwd)/venv/lib/python3.10/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH"

#1. run the API
python /home/api/VLLM_api/VLLM_API.py

# or run both 0-1.
venv/bin/python VLLM_API.py

# Specific GPU
CUDA_VISIBLE_DEVICES=0 ./venv/bin/python VLLM_API.py

# Run in background
CUDA_VISIBLE_DEVICES=0 nohup ./venv/bin/python -u VLLM_API.py > log.log 2>&1 &