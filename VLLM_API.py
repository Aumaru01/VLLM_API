import os
import json
import time
import yaml
import hashlib
import asyncio
import contextlib

from vllm import LLM
from pathlib import Path
from typing import Optional
from transformers import AutoTokenizer
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from utils.schema import (
    UniTextItem,
    MultiTextItem,
    GenerateStructuredRequest,
    GenerateBatchStructuredRequest,
)
from utils.run_job import JobRunner

os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

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

RESULT_DIR = Path(_cfg["result_dir"])
RESULT_DIR.mkdir(exist_ok=True)


# ===========================================================
# Load Functions
# ============================================================
def _save_result(task_id: str, data: dict) -> None:
    (RESULT_DIR / f"{task_id}.json").write_text(json.dumps(data, ensure_ascii=False))

def _load_result(task_id: str) -> dict | None:
    path = RESULT_DIR / f"{task_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None

def _make_id(prefix: str) -> str:
    raw = str(time.time_ns())
    return f"{prefix}_{hashlib.shake_128(raw.encode()).hexdigest(8)}"

# ============================================================
# Global state
# ============================================================
MODEL_STATE: dict = {}
task_store: dict[str, dict] = {}
queue: asyncio.Queue = asyncio.Queue()

# ============================================================
# Inference jobs (run off the event loop via asyncio.to_thread)
# ============================================================
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
    JOBRUNNER = JobRunner(_cfg, MODEL_STATE)
    MODEL_STATE["run_general_single"] = JOBRUNNER.run_general_single
    MODEL_STATE["run_general_batch"] = JOBRUNNER.run_general_batch
    MODEL_STATE["run_general_structured"] = JOBRUNNER.run_general_structured
    MODEL_STATE["run_general_batch_structured"] = JOBRUNNER.run_general_batch_structured
    MODEL_STATE["run_sentiment_single"] = JOBRUNNER.run_sentiment_single
    MODEL_STATE["run_sentiment_batch"] = JOBRUNNER.run_sentiment_batch
    
    print("[Startup] Model loaded successfully.")
    worker_task = asyncio.create_task(_worker())
    
    yield
    worker_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await worker_task
    MODEL_STATE.clear()


async def _worker() -> None:
    while True:
        job = await queue.get()
        task_id = job["task_id"]
        task_store[task_id]["status"] = "running"
        try:
            task_type = job["task_type"]
            if task_type == "general_single":
                handler_result = await asyncio.to_thread(
                    MODEL_STATE["run_general_single"], 
                    job["text"], job["max_tokens"], job["temperature"])
            elif task_type == "general_batch":
                handler_result = await asyncio.to_thread(
                    MODEL_STATE["run_general_batch"], 
                    job["ids"], job["texts"], job["max_tokens"], job["temperature"])
            elif task_type == "sentiment_single":
                handler_result = await asyncio.to_thread(
                    MODEL_STATE["run_sentiment_single"], 
                    job["text"], job["max_tokens"], job["temperature"])
            elif task_type == "sentiment_batch":
                handler_result = await asyncio.to_thread(
                    MODEL_STATE["run_sentiment_batch"], 
                    job["ids"], job["texts"], job["max_tokens"], job["temperature"])
            elif task_type == "general_structured":
                handler_result = await asyncio.to_thread(
                    MODEL_STATE["run_general_structured"], 
                    job["text"], job["json_schema"], job["max_tokens"], job["temperature"])
            elif task_type == "general_batch_structured":
                handler_result = await asyncio.to_thread(
                    MODEL_STATE["run_general_batch_structured"],
                    job["ids"], job["texts"], job["json_schema"], job["max_tokens"], job["temperature"])
            else:
                raise ValueError(f"Unknown task_type: {task_type}")
            
            time_used_ms = handler_result.pop("time_usage_ms")
            token_usage = handler_result.pop("token_usage")
            result = handler_result["result"] if list(handler_result.keys()) == ["result"] else handler_result
            task_store[task_id] = {
                "status": "done",
                "time_used_ms": time_used_ms,
                "token_usage": token_usage,
                "result": result,
            }
        except Exception as e:
            task_store[task_id] = {"status": "error", "error": str(e)}
        _save_result(task_id, task_store[task_id])
        queue.task_done()


def _enqueue(task_type: str, **job_fields) -> str:
    task_id = _make_id(task_type)
    task_store[task_id] = {"status": "queued"}
    queue.put_nowait({"task_id": task_id, "task_type": task_type, **job_fields})
    return task_id


app = FastAPI(
    title="vLLM Inference API",
    description="Generic LLM inference API — accepts chat messages, returns generated text",
    lifespan=lifespan,
)

# ============================================================
# Endpoints
# ============================================================
@app.post("/general", summary="Queue generation of general text from a single chat text", status_code=202)
async def General(
    req: UniTextItem,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
):
    if "llm" not in MODEL_STATE:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    task_id = _enqueue(
        "general_single", 
        text=req.text, 
        max_tokens=max_tokens, 
        temperature=temperature
    )
    return {"task_id": task_id, "status": "queued"}


@app.post("/general_batch", summary="Queue generation of general text from multiple chat texts in a single batch", status_code=202,)
async def General_batch(
    req: list[MultiTextItem],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
):
    if "llm" not in MODEL_STATE:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    ids = [item.id for item in req]
    texts = [item.text for item in req]
    
    if not texts:
        raise HTTPException(status_code=422, detail="texts must not be empty")
    
    if len(set(ids)) != len(ids):
        raise HTTPException(status_code=422, detail="Duplicate ids found in texts")

    task_id = _enqueue(
        "general_batch", 
        ids=ids, 
        texts=texts, 
        max_tokens=max_tokens, 
        temperature=temperature
    )
    return {"task_id": task_id, "status": "queued"}


@app.post("/general_structured", summary="Queue generation of general structured JSON output from a single text", status_code=202,)
async def General_structured(
    req: GenerateStructuredRequest,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
):
    if "llm" not in MODEL_STATE:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    task_id = _enqueue(
        "general_structured",
        text=req.text.text,
        json_schema=req.json_schema,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return {"task_id": task_id, "status": "queued"}


@app.post("/general_batch_structured", summary="Queue generation of general structured JSON output from multiple texts", status_code=202,)
async def General_batch_structured(
    req: GenerateBatchStructuredRequest,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
):
    if "llm" not in MODEL_STATE:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    ids = [item.id for item in req.texts]
    texts = [item.text for item in req.texts]

    if not texts:
        raise HTTPException(status_code=422, detail="texts must not be empty")

    if len(set(ids)) != len(ids):
        raise HTTPException(status_code=422, detail="Duplicate ids found in texts")

    task_id = _enqueue(
        "general_batch_structured",
        ids=ids,
        texts=texts,
        json_schema=req.json_schema,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return {"task_id": task_id, "status": "queued"}


@app.post("/sentiment", summary="Queue sentiment analysis of a single chat text", status_code=202)
async def sentiment(
    req: UniTextItem,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
):
    if "llm" not in MODEL_STATE:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    task_id = _enqueue(
        "sentiment_single", 
        text=req.text, 
        max_tokens=max_tokens, 
        temperature=temperature
    )
    return {"task_id": task_id, "status": "queued"}

@app.post("/sentiment_batch", summary="Queue sentiment analysis of multiple chat texts", status_code=202)
async def sentiment_batch(
    req: list[MultiTextItem],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
):
    if "llm" not in MODEL_STATE:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    ids = [item.id for item in req]
    texts = [item.text for item in req]

    if not texts:
        raise HTTPException(status_code=422, detail="texts must not be empty")

    if len(set(ids)) != len(ids):
        raise HTTPException(status_code=422, detail="Duplicate ids found in texts")

    task_id = _enqueue(
        "sentiment_batch",
        ids=ids,
        texts=texts,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return {"task_id": task_id, "status": "queued"}


@app.get("/result/{task_id}", summary="ดึงสถานะ/ผลลัพธ์ของ task จาก task_id")
async def get_result(task_id: str):
    task = task_store.get(task_id)
    if task is not None:
        return {"task_id": task_id, **task}

    saved = _load_result(task_id)
    if saved is not None:
        return {"task_id": task_id, **saved}

    raise HTTPException(status_code=404, detail="task_id not found")


@app.get("/health", summary="ตรวจสอบสถานะ API")
async def health():
    return {"status": "ok", "model_loaded": "llm" in MODEL_STATE, "queue_size": queue.qsize()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("VLLM_API:app", host=SERVER_HOST, port=SERVER_PORT, reload=False)
