#0 if use environment(venv)
source /venv/bin/activate

#1. run the API
python /home/api/VLLM_api/VLLM_API.py

# or run both 0-1.
venv/bin/python VLLM_API.py

# Specific GPU
CUDA_VISIBLE_DEVICES=0 venv/bin/python VLLM_API.py

# Run in background
CUDA_VISIBLE_DEVICES=0 nohup venv/bin/python VLLM_API.py > log.log 2>&1 &