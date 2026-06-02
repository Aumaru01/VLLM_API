import os
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from pydantic import BaseModel

import yaml
from fastapi import FastAPI, HTTPException
from vllm import LLM, SamplingParams
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
    messages: list[dict[str, str]]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class GenerateBatchRequest(BaseModel):
    prompts: list[list[dict[str, str]]]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


# ============================================================
# Endpoints
# ============================================================
@app.post("/generate", summary="Generate text from a single chat prompt")
async def generate(req: GenerateRequest):
    if "llm" not in MODEL_STATE:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    tokenizer = MODEL_STATE["tokenizer"]
    prompt = tokenizer.apply_chat_template(req.messages, tokenize=False, add_generation_prompt=True)
    params = SamplingParams(
        temperature=req.temperature if req.temperature is not None else DEFAULT_TEMPERATURE,
        max_tokens=req.max_tokens if req.max_tokens is not None else DEFAULT_MAX_TOKENS,
    )
    outputs = MODEL_STATE["llm"].generate([prompt], params)
    return {"text": outputs[0].outputs[0].text.strip()}


@app.post("/generate_batch", summary="Generate text from multiple chat prompts in a single batch")
async def generate_batch(req: GenerateBatchRequest):
    if "llm" not in MODEL_STATE:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not req.prompts:
        raise HTTPException(status_code=422, detail="prompts must not be empty")

    tokenizer = MODEL_STATE["tokenizer"]
    prompts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in req.prompts
    ]
    params = SamplingParams(
        temperature=req.temperature if req.temperature is not None else DEFAULT_TEMPERATURE,
        max_tokens=req.max_tokens if req.max_tokens is not None else DEFAULT_MAX_TOKENS,
    )
    outputs = MODEL_STATE["llm"].generate(prompts, params)
    return {"texts": [o.outputs[0].text.strip() for o in outputs]}


@app.get("/health", summary="ตรวจสอบสถานะ API")
async def health():
    return {"status": "ok", "model_loaded": "llm" in MODEL_STATE}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=SERVER_HOST, port=SERVER_PORT, reload=False)
