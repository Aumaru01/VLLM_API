# VLLM API

A FastAPI service that wraps [vLLM](https://github.com/vllm-project/vllm) for self-hosted LLM inference. It exposes an async, queue-based HTTP API for general text generation, structured (JSON-schema constrained) generation, Thai-language sentiment analysis, Thai NER, and Excel-transcript (VTT) summarization — all served from a single locally loaded model.

## Features

- Single vLLM model instance loaded once at startup, shared across all requests
- Async job queue: requests return a `task_id` immediately (`202 Accepted`), results are fetched by polling
- Batch endpoints for processing multiple texts in one vLLM call
- Structured output generation constrained to a caller-supplied JSON schema
- PDF upload endpoint: renders pages to images and sends them to the model as multimodal input (falls back to a "model not support upload file" note if the loaded model isn't vision-capable)
- Built-in Thai sentiment analysis prompt (Positive / Neutral / Negative)
- Thai NER: sentence-tokenizes input, extracts entities per sentence, and aggregates/filters them by frequency
- VTT summary: takes an uploaded Excel transcript (`Date`, `Time`, `Text`, `TextID` columns), groups it into same-context news/ads/general segments, and summarizes each
- Results persisted to disk (`result_dir`) so they survive process restarts
- Config-driven via `config.yaml` (server, model, inference defaults, sentiment/ner/vtt_summuary settings)

## Requirements

- Python 3.10+
- An NVIDIA GPU supported by vLLM
- Dependencies in `requirements.txt`:
  - `vllm`, `fastapi`, `pydantic`, `transformers`, `PyYAML`, `huggingface_hub`, `pypdfium2` (PDF page rendering), `pythainlp` (Thai sentence tokenization for NER), `pandas`/`openpyxl` (Excel parsing for VTT summary)

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
| `model` | `dtype` | Compute/weights precision passed to vLLM:<br>• `"auto"`: Uses the precision specified in the model's own config (safest default, matches how the checkpoint was trained).<br>• `"bfloat16"`: Same memory footprint as fp16 but wider dynamic range; the recommended choice on Ampere/Hopper+ GPUs and most modern LLM checkpoints.<br>• `"float16"` / `"half"`: Standard half precision; slightly smaller dynamic range than bf16, which can occasionally cause numerical instability (overflow/NaN) on some models.<br>• `"float32"`: Full precision; ~2x the memory and compute cost of the 16-bit options, only useful for accuracy debugging, not routine inference. |
| `model` | `gpu_memory_utilization` | Fraction of GPU memory vLLM may use |
| `model` | `quantization` | Model quantization options:<br>• `None`: Not quantization.<br>• `"fp8"`: Forces BF16 to FP8 in real-time, or loads a true FP8 model (reduces RAM by 50% while maintaining almost 100% intelligence).<br>• `"awq"`: For 4-bit AWQ family models (maximum RAM efficiency, very fast on vLLM).<br>• `"gptq"`: For native GPTQ 4-bit or 8-bit models. <br>• `Other`: https://docs.vllm.ai/en/latest/features/quantization/|
| `model` | `trust_remote_code` | Passed to vLLM/tokenizer for custom model code |
| `model` | `enforce_eager` | Passed to vLLM. `true` disables CUDA graph capture, lowering upfront VRAM usage at some cost to throughput; `false` lets vLLM capture graphs for faster steady-state inference |
| `inference` | `default_max_tokens` | Fallback `max_tokens` used when a request omits it |
| `inference` | `default_temperature` | Fallback sampling param when a request omits it |
| `inference` | `seed` | Sampling seed used for all generations |
| `sentiment` | `max_string_length` | Characters of input text used for sentiment analysis |
| `sentiment` | `only_sentiment_output` | If true, forces output to one of Positive/Neutral/Negative |
| `ner` | `ner_tag` | Comma-separated list of entity tags the NER prompt asks the model to extract |
| `ner` | `minimum_count` | An extracted entity must appear more than this many times (across a text's sentences) to be kept in the output |
| `vtt_summuary` | `vtt_summary_req_col` | Column names required in the uploaded VTT Excel file, e.g. `["Date", "Time", "Text", "TextID"]` |
| `vtt_summuary` | `system_instruction_path` | Path to the `VTT_Summary_system_instruction/` directory that system-instruction `.txt` files are loaded from. Four variants ship there: `system_instruction.txt` (default, groups rows into `same_context_text` plus a `summary_text`), `system_instruction_only_summarize.txt` (drops the raw grouped text, returns only the summary), and `_acc_mode` variants of each that additionally return `alltext_in_one` (all row texts concatenated with `\|\|`) for traceability. Which file is used per request is chosen via the `system_instruction` query param on `POST /VTT_summary_single`, not this config key |
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
- Query params: `max_tokens` (optional, falls back to `config.yaml`'s `default_max_tokens`), `temperature` (optional, falls back to `config.yaml`'s `default_temperature`)

### `POST /general_batch`
Queue generation for multiple texts in one batch.
- Body: `[{"id": "...", "text": "..."}, ...]`

### `POST /general_structured`
Queue generation constrained to a JSON schema.
- Body: `{"text": {"text": "..."}, "json_schema": {...}}`

### `POST /general_batch_structured`
Batch version of structured generation.
- Body: `{"texts": [{"id": "...", "text": "..."}, ...], "json_schema": {...}}`

### `POST /general_with_PDF`
Queue generation from a chat text plus an uploaded PDF file. Multipart form (`multipart/form-data`), not JSON:
- `file`: the PDF (required)
- `text`: prompt text (optional form field)
- `max_tokens`, `temperature` (optional query params)

Each page of the PDF is rendered to an image (up to 8 pages) and sent to the model as multimodal input via `LLM.chat()`. If the loaded model isn't vision-capable (or the file fails to render), the result comes back with `"note": "model not support upload file"` and an `error` field instead of raising.

### `POST /sentiment`
Queue Thai sentiment analysis for a single text.
- Body: `{"text": "..."}`
- Returns (once done) a result with `raw_text`, truncated `text`, and `sentiment`

### `POST /sentiment_batch`
Batch version of sentiment analysis.
- Body: `[{"id": "...", "text": "..."}, ...]`

### `POST /ner`
Queue Thai named-entity extraction for a single text.
- Body: `{"text": "..."}`
- Input is sentence-tokenized (`pythainlp`), each sentence is run through the NER prompt, and matching entities are aggregated across sentences, keeping only those seen more than `ner.minimum_count` times.
- Returns (once done) a result with `raw_text`, cleaned `text`, and `output` (entities grouped by tag with their counts)

### `POST /ner_batch`
Batch version of NER extraction.
- Body: `[{"id": "...", "text": "..."}, ...]`

### `POST /VTT_summary_single`
Queue VTT (transcript) summarization from an uploaded Excel file. Multipart form (`multipart/form-data`), not JSON:
- `file`: an `.xlsx` file with `Date`, `Time`, `Text`, `TextID` columns (required; column names are configurable via `vtt_summuary.vtt_summary_req_col`)
- `max_tokens`, `temperature` (optional query params)
- `system_instruction` (optional query param, default `"system_instruction.txt"`): filename of the prompt to load from `vtt_summuary.system_instruction_path`, e.g. `system_instruction_only_summarize.txt`, `system_instruction_acc_mode.txt`, `system_instruction_only_summarize_acc_mode.txt` (the `.txt` suffix is optional)

The rows are cleaned and converted to JSON records, appended to the chosen system instruction file, and sent to the model as a single prompt. The model groups same-context rows, classifies each group (News/Ads/General, with subtypes), and returns a summary per group.

### `GET /result/{task_id}`
Fetch the status/result of a queued task. Response includes `status` (`queued`, `running`, `done`, `error`), and once done, `time_used_ms`, `token_usage`, and `result`.

### `GET /health`
Returns `{"status": "ok", "model_loaded": bool, "queue_size": int}`.

## Notes

- Duplicate `id` values in batch requests are rejected with `422`.
- Requests made before the model finishes loading are rejected with `503`.
- Completed task results are cached in memory and also persisted to `result_dir/<task_id>.json`, so `GET /result` still works after a restart as long as the file exists.
