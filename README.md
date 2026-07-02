# VLLM API

A FastAPI service that wraps [vLLM](https://github.com/vllm-project/vllm) for self-hosted LLM inference. It exposes an async, queue-based HTTP API for general text generation, structured (JSON-schema constrained) generation, and Thai-language sentiment analysis — all served from a single locally loaded model.

## Features

- Single vLLM model instance loaded once at startup, shared across all requests
- Async job queue: requests return a `task_id` immediately (`202 Accepted`), results are fetched by polling
- Batch endpoints for processing multiple texts in one vLLM call
- Structured output generation constrained to a caller-supplied JSON schema
- Built-in Thai sentiment analysis prompt (Positive / Neutral / Negative)
- Results persisted to disk (`result_dir`) so they survive process restarts
- Config-driven via `config.yaml` (server, model, inference defaults, sentiment settings)

## Requirements

- Python 3.10+
- An NVIDIA GPU supported by vLLM
- Dependencies in `requirements.txt`:
  - `vllm`, `fastapi`, `pydantic`, `transformers`, `PyYAML`, `huggingface_hub`

## Setup

1. Create/activate a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Copy the example config and edit it for your environment:

   ```bash
   cp config.yaml.example config.yaml
   ```

   Set `model.path` to either a local model directory or a Hugging Face Hub repo id, and set `result_dir` to a writable path where task results will be saved.

3. (Optional) Download a model from Hugging Face Hub ahead of time with `hf_model_downloader.py` — edit `model_repo`, `save_dir`, and `hf_token` at the top of the file, then run:

   ```bash
   python hf_model_downloader.py
   ```

## Configuration (`config.yaml`)

| Section | Key | Description |
|---|---|---|
| `server` | `host`, `port` | Address the FastAPI/uvicorn server binds to |
| `model` | `path` | Local path or HF Hub id of the model to load |
| `model` | `dtype` | vLLM dtype (e.g. `bfloat16`) |
| `model` | `gpu_memory_utilization` | Fraction of GPU memory vLLM may use |
| `model` | `trust_remote_code` | Passed to vLLM/tokenizer for custom model code |
| `inference` | `default_max_tokens`, `default_temperature` | Fallback sampling params when a request omits them |
| `inference` | `seed` | Sampling seed used for all generations |
| `sentiment` | `max_string_length` | Characters of input text used for sentiment analysis |
| `sentiment` | `only_sentiment_output` | If true, forces output to one of Positive/Neutral/Negative |
| `result_dir` | — | Directory where completed task results are written as JSON |

## Running

```bash
# activate venv, then:
python VLLM_API.py

# or restrict to a specific GPU:
CUDA_VISIBLE_DEVICES=0 python VLLM_API.py

# run in the background, logging to log.log:
CUDA_VISIBLE_DEVICES=0 nohup venv/bin/python VLLM_API.py > log.log 2>&1 &
```

See `run.sh` for these variants. The server loads the model at startup (this can take a while for large models) before it starts accepting requests.

## API

All generation/sentiment endpoints are async: they enqueue a job and return a `task_id` immediately. Poll `GET /result/{task_id}` until `status` is `done` (or `error`).

### `POST /general`
Queue single-text generation.
- Body: `{"text": "..."}`
- Query params: `max_tokens`, `temperature` (optional, fall back to config defaults)

### `POST /general_batch`
Queue generation for multiple texts in one batch.
- Body: `[{"id": "...", "text": "..."}, ...]`

### `POST /general_structured`
Queue generation constrained to a JSON schema.
- Body: `{"text": {"text": "..."}, "json_schema": {...}}`

### `POST /general_batch_structured`
Batch version of structured generation.
- Body: `{"texts": [{"id": "...", "text": "..."}, ...], "json_schema": {...}}`

### `POST /sentiment`
Queue Thai sentiment analysis for a single text.
- Body: `{"text": "..."}`
- Returns (once done) a result with `raw_text`, truncated `text`, and `sentiment`

### `POST /sentiment_batch`
Batch version of sentiment analysis.
- Body: `[{"id": "...", "text": "..."}, ...]`

### `GET /result/{task_id}`
Fetch the status/result of a queued task. Response includes `status` (`queued`, `running`, `done`, `error`), and once done, `time_used_ms`, `token_usage`, and `result`.

### `GET /health`
Returns `{"status": "ok", "model_loaded": bool, "queue_size": int}`.

## Notes

- Duplicate `id` values in batch requests are rejected with `422`.
- Requests made before the model finishes loading are rejected with `503`.
- Completed task results are cached in memory and also persisted to `result_dir/<task_id>.json`, so `GET /result` still works after a restart as long as the file exists.
