import os
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from pydantic import BaseModel

import json
import time

import yaml
from fastapi import FastAPI, HTTPException
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from transformers import AutoTokenizer

# ============================================================
# Config
# ============================================================
_cfg_path = Path(__file__).parent / "config.yaml"
with open(_cfg_path) as f:
    _cfg = yaml.safe_load(f)

SERVER_HOST: str = _cfg["server"]["host"]
SERVER_PORT: int = _cfg["server"]["port"]

MODEL_PATH: str = _cfg["model"]["path"]
MODEL_DTYPE: str = _cfg["model"]["dtype"]
MODEL_GPU_UTIL: float = _cfg["model"]["gpu_memory_utilization"]
MODEL_TRUST_REMOTE: bool = _cfg["model"]["trust_remote_code"]

DEFAULT_MAX_TOKENS: int = _cfg["inference"]["default_max_tokens"]
DEFAULT_TEMPERATURE: float = _cfg["inference"]["default_temperature"]
SEED: int = _cfg["inference"]["seed"]

# ============================================================
# Global state
# ============================================================
MODEL_STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[Startup] Loading vLLM model: {MODEL_PATH}")
    llm = LLM(
        model=MODEL_PATH,
        dtype=MODEL_DTYPE,
        gpu_memory_utilization=MODEL_GPU_UTIL,
        trust_remote_code=MODEL_TRUST_REMOTE,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=MODEL_TRUST_REMOTE)
    MODEL_STATE["llm"] = llm
    MODEL_STATE["tokenizer"] = tokenizer
    print("[Startup] Model loaded successfully.")
    yield
    MODEL_STATE.clear()


app = FastAPI(
    title="vLLM Inference API",
    description="Generic LLM inference API — accepts chat messages, returns generated text",
    lifespan=lifespan,
)


# ============================================================
# Request models
# ============================================================
class GenerateRequest(BaseModel):
    prompt: str


class GenerateBatchRequest(BaseModel):
    prompts: list[str]


class GenerateStructuredRequest(BaseModel):
    prompt: str
    json_schema: dict


class GenerateBatchStructuredRequest(BaseModel):
    prompts: list[str]
    json_schema: dict


# ============================================================
# Endpoints
# ============================================================
@app.post("/generate", summary="Generate text from a single chat prompt")
async def generate(
    req: GenerateRequest,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
):
    if "llm" not in MODEL_STATE:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    tokenizer = MODEL_STATE["tokenizer"]
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": req.prompt}], tokenize=False, add_generation_prompt=True
    )
    params = SamplingParams(
        temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS,
        seed = SEED
    )
    start = time.time()
    outputs = MODEL_STATE["llm"].generate([prompt], params)
    stop = time.time()
    prompt_tokens = len(outputs[0].prompt_token_ids)
    completion_tokens = len(outputs[0].outputs[0].token_ids)
    return {
            "text": outputs[0].outputs[0].text.strip(),
            "time_usage_ms": (stop-start) * 1000,
            "token_usage": {
                "prompt_bytes": len(req.prompt.encode()),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            }


@app.post("/generate_batch", summary="Generate text from multiple chat prompts in a single batch")
async def generate_batch(
    req: GenerateBatchRequest,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
):
    if "llm" not in MODEL_STATE:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not req.prompts:
        raise HTTPException(status_code=422, detail="prompts must not be empty")

    tokenizer = MODEL_STATE["tokenizer"]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True
        )
        for msg in req.prompts
    ]
    params = SamplingParams(
        temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS,
        seed = SEED
    )
    start = time.time()
    outputs = MODEL_STATE["llm"].generate(prompts, params)
    stop = time.time()
    prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    completion_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    return {
            "texts": [o.outputs[0].text.strip() for o in outputs],
            "time_usage_ms": (stop-start) * 1000,
            "token_usage": {
                "prompt_bytes": sum(len(p.encode()) for p in req.prompts),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            }


@app.post("/generate_structured", summary="Generate structured JSON output from a single prompt")
async def generate_structured(
    req: GenerateStructuredRequest,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
):
    if "llm" not in MODEL_STATE:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    tokenizer = MODEL_STATE["tokenizer"]
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": req.prompt}], tokenize=False, add_generation_prompt=True
    )
    params = SamplingParams(
        temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS,
        structured_outputs=StructuredOutputsParams(json=req.json_schema),
        seed = SEED
    )
    start = time.time()
    outputs = MODEL_STATE["llm"].generate([prompt], params)
    stop = time.time()
    raw = outputs[0].outputs[0].text.strip()
    prompt_tokens = len(outputs[0].prompt_token_ids)
    completion_tokens = len(outputs[0].outputs[0].token_ids)
    token_usage = {
        "prompt_bytes": len(req.prompt.encode()),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    try:
        return {
                "result": json.loads(raw),
                "time_usage_ms": (stop-start) * 1000,
                "token_usage": token_usage,
                }
    except json.JSONDecodeError:
        return {
                "result": raw,
                "time_usage_ms": (stop-start) * 1000,
                "token_usage": token_usage,
                }


@app.post("/generate_batch_structured", summary="Generate structured JSON output from multiple prompts")
async def generate_batch_structured(
    req: GenerateBatchStructuredRequest,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
):
    if "llm" not in MODEL_STATE:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not req.prompts:
        raise HTTPException(status_code=422, detail="prompts must not be empty")

    tokenizer = MODEL_STATE["tokenizer"]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True
        )
        for msg in req.prompts
    ]
    params = SamplingParams(
        temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS,
        structured_outputs=StructuredOutputsParams(json=req.json_schema),
        seed = SEED
    )
    start = time.time()
    outputs = MODEL_STATE["llm"].generate(prompts, params)
    stop = time.time()
    results = []
    for o in outputs:
        raw = o.outputs[0].text.strip()
        try:
            results.append(json.loads(raw))
        except json.JSONDecodeError:
            results.append(raw)
    prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    completion_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    return {
            "results": results,
            "time_usage_ms": (stop-start) * 1000,
            "token_usage": {
                "prompt_bytes": sum(len(p.encode()) for p in req.prompts),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            }


@app.get("/health", summary="ตรวจสอบสถานะ API")
async def health():
    return {"status": "ok", "model_loaded": "llm" in MODEL_STATE}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("VLLM_API:app", host=SERVER_HOST, port=SERVER_PORT, reload=False)
