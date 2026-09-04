import os
import time
import asyncio
import contextlib

from vllm import LLM
from typing import Optional
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, File, Form, UploadFile

from __init__ import (
    CFG,
    SERVER_HOST,
    SERVER_PORT,
    MODEL_PATH,
    QUANTIZATION,
    MODEL_DTYPE,
    MODEL_GPU_UTIL,
    MODEL_TRUST_REMOTE,
    DEFAULT_TEMPERATURE,
    ENFORCE_EAGER
)
from utils.schema import (
    UniTextItem,
    MultiTextItem,
    GenerateStructuredRequest,
    GenerateBatchStructuredRequest,
)
from utils.run_job import JobRunner
from utils.funtion import _save_result, _load_result, _make_id, _scan_disk_tasks

os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

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
        quantization = QUANTIZATION,
        dtype=MODEL_DTYPE,
        gpu_memory_utilization=MODEL_GPU_UTIL,
        trust_remote_code=MODEL_TRUST_REMOTE,
        enforce_eager=ENFORCE_EAGER ## ลดการกิน VRAM ล่วงหน้า
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=MODEL_TRUST_REMOTE)
    MODEL_STATE["llm"] = llm
    MODEL_STATE["tokenizer"] = tokenizer
    JOBRUNNER = JobRunner(CFG, MODEL_STATE)
    MODEL_STATE["run_general_single"] = JOBRUNNER.run_general_single
    MODEL_STATE["run_general_batch"] = JOBRUNNER.run_general_batch
    MODEL_STATE["run_general_structured"] = JOBRUNNER.run_general_structured
    MODEL_STATE["run_general_batch_structured"] = JOBRUNNER.run_general_batch_structured
    MODEL_STATE["run_general_with_PDF"] = JOBRUNNER.run_general_with_PDF
    MODEL_STATE["run_sentiment_single"] = JOBRUNNER.run_sentiment_single
    MODEL_STATE["run_sentiment_batch"] = JOBRUNNER.run_sentiment_batch
    MODEL_STATE["run_ner_single"] = JOBRUNNER.run_ner_single
    MODEL_STATE["run_ner_batch"] = JOBRUNNER.run_ner_batch
    MODEL_STATE["run_VTT_summary_single"] = JOBRUNNER.run_VTT_summary_single
    
    print("[Startup] Model loaded successfully.")
    worker_task = asyncio.create_task(_worker())
    
    yield
    worker_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await worker_task
    MODEL_STATE.clear()

def _check_expired_files(cutoff_days: int = 30):
    try:
        RESULT_DIR = Path("./result")
        cutoff_time = time.time() - (cutoff_days * 86400)

        for file_path in RESULT_DIR.rglob("*"):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()  # Deletes file in pathlib
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
    except Exception as e:
        print(f"Directory processing error: {e}")


async def _worker() -> None:
    while True:
        job = await queue.get()
        task_id = job["task_id"]
        task_store[task_id]["status"] = "running"
        try:
            _check_expired_files()  # Check and delete expired files before processing the job
            
            task_type = job["task_type"]
            
            if task_type == "general_single":
                handler_result = await asyncio.to_thread(
                    MODEL_STATE["run_general_single"], 
                    job["text"], job["max_tokens"], job["temperature"])
            
            elif task_type == "general_batch":
                handler_result = await asyncio.to_thread(
                    MODEL_STATE["run_general_batch"], 
                    job["ids"], job["texts"], job["max_tokens"], job["temperature"])
            
            elif task_type == "general_structured":
                handler_result = await asyncio.to_thread(
                    MODEL_STATE["run_general_structured"], 
                    job["text"], job["json_schema"], job["max_tokens"], job["temperature"])
            
            elif task_type == "general_batch_structured":
                handler_result = await asyncio.to_thread(
                    MODEL_STATE["run_general_batch_structured"],
                    job["ids"], job["texts"], job["json_schema"], job["max_tokens"], job["temperature"])
            
            elif task_type == "sentiment_single":
                handler_result = await asyncio.to_thread(
                    MODEL_STATE["run_sentiment_single"],
                    job["text"], job["max_tokens"], job["temperature"], job["prompt_file"])

            elif task_type == "sentiment_batch":
                handler_result = await asyncio.to_thread(
                    MODEL_STATE["run_sentiment_batch"],
                    job["ids"], job["texts"], job["max_tokens"], job["temperature"], job["prompt_file"])
                
            elif task_type == "ner_single":
                handler_result = await asyncio.to_thread(
                    MODEL_STATE["run_ner_single"], 
                    job["text"], job["max_tokens"], job["temperature"])
            
            elif task_type == "ner_batch":
                handler_result = await asyncio.to_thread(
                    MODEL_STATE["run_ner_batch"], 
                    job["ids"], job["texts"], job["max_tokens"], job["temperature"])
            
            elif task_type == "general_with_PDF":
                handler_result = await asyncio.to_thread(
                    MODEL_STATE["run_general_with_PDF"],
                    job["text"], job["file_bytes"], job["filename"], job["max_tokens"], job["temperature"])
            
            elif task_type == "VTT_summary_single":
                handler_result = await asyncio.to_thread(
                    MODEL_STATE["run_VTT_summary_single"],
                    job["file_bytes"], job["max_tokens"], job["temperature"], job["system_instruction"])
            
            else:
                raise ValueError(f"Unknown task_type: {task_type}")
            
            time_used_ms = handler_result.pop("time_usage_ms")
            token_usage = handler_result.pop("token_usage")
            result = handler_result["result"] if list(handler_result.keys()) == ["result"] else handler_result
            task_store[task_id] = {
                **task_store[task_id],
                "status": "done",
                "time_used_ms": time_used_ms,
                "token_usage": token_usage,
                "result": result,
            }
        except Exception as e:
            task_store[task_id] = {**task_store[task_id], "status": "error", "error": str(e)}
        _save_result(task_id, task_store[task_id])
        queue.task_done()


def _enqueue(task_type: str, **job_fields) -> str:
    task_id = _make_id(task_type)
    task_store[task_id] = {"status": "queued", "created_at": time.time()}
    queue.put_nowait({"task_id": task_id, "task_type": task_type, **job_fields})
    return task_id


app = FastAPI(
    title="vLLM Inference API",
    description=f"Generic LLM inference API — accepts chat messages, returns generated text\n\n model path: {MODEL_PATH}",
    lifespan=lifespan,
)

# ============================================================
# Endpoints
# ============================================================
@app.post("/general", summary="Queue generation of general text from a single chat text", status_code=202)
async def General(
    req: UniTextItem,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = DEFAULT_TEMPERATURE,
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
    temperature: Optional[float] = DEFAULT_TEMPERATURE,
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
    temperature: Optional[float] = DEFAULT_TEMPERATURE,
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
    temperature: Optional[float] = DEFAULT_TEMPERATURE,
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


@app.post("/general_with_PDF", summary="Queue generation of general text from a chat text with an uploaded PDF file", status_code=202)
async def General_with_file(
    file: UploadFile = File(...),
    text: Optional[str] = Form(None),
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = DEFAULT_TEMPERATURE,
):
    if "llm" not in MODEL_STATE:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if file.content_type != "application/pdf" and not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are supported")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")

    task_id = _enqueue(
        "general_with_PDF",
        text=text,
        file_bytes=file_bytes,
        filename=file.filename,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return {"task_id": task_id, "status": "queued"}


@app.post("/sentiment", summary="Queue sentiment analysis of a single chat text", status_code=202)
async def sentiment(
    req: UniTextItem,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = DEFAULT_TEMPERATURE,
    prompt_file: str = "default.txt",
):
    if "llm" not in MODEL_STATE:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    task_id = _enqueue(
        "sentiment_single",
        text=req.text,
        max_tokens=max_tokens,
        temperature=temperature,
        prompt_file=prompt_file,
    )
    return {"task_id": task_id, "status": "queued"}


@app.post("/sentiment_batch", summary="Queue sentiment analysis of multiple chat texts", status_code=202)
async def sentiment_batch(
    req: list[MultiTextItem],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = DEFAULT_TEMPERATURE,
    prompt_file: str = "default.txt",
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
        prompt_file=prompt_file,
    )
    return {"task_id": task_id, "status": "queued"}


@app.post("/ner", summary="Queue text ner of a single chat text", status_code=202)
async def ner(
    req: UniTextItem,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = DEFAULT_TEMPERATURE,
):
    if "llm" not in MODEL_STATE:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    task_id = _enqueue(
        "ner_single", 
        text=req.text, 
        max_tokens=max_tokens, 
        temperature=temperature
    )
    return {"task_id": task_id, "status": "queued"}


@app.post("/ner_batch", summary="Queue text ner of a single chat text", status_code=202)
async def ner_batch(
    req: list[MultiTextItem],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = DEFAULT_TEMPERATURE,
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
        "ner_batch",
        ids=ids,
        texts=texts,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return {"task_id": task_id, "status": "queued"}


@app.post("/VTT_summary_single", summary="Queue generation of VTT summary from an uploaded Excel file", status_code=202)
async def General_with_file(
    file: UploadFile = File(...),
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = DEFAULT_TEMPERATURE,
    system_instruction: str = "system_instruction.txt",
):
    if "llm" not in MODEL_STATE:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if file.content_type != "application/excel" and not (file.filename or "").lower().endswith(".xlsx"):
        raise HTTPException(status_code=422, detail="Only Excel files are supported")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")

    task_id = _enqueue(
        "VTT_summary_single",
        file_bytes=file_bytes,
        max_tokens=max_tokens,
        temperature=temperature,
        system_instruction=system_instruction,
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


@app.get("/queue", summary="List task ids and status from the current run plus disk history")
async def list_queue(days: int = 3):
    cutoff = time.time() - days * 86400
    tasks = _scan_disk_tasks(cutoff)

    for tid, t in task_store.items():
        created_at = t.get("created_at", time.time())
        if created_at >= cutoff:
            tasks[tid] = {"status": t.get("status"), "created_at": created_at}
        else:
            tasks.pop(tid, None)

    ordered = sorted(tasks.items(), key=lambda kv: kv[1]["created_at"], reverse=True)
    return {
        "queue_size": queue.qsize(),
        "count": len(ordered),
        "tasks": [
            {
                "task_id": tid,
                "status": info["status"],
                "created_at": datetime.fromtimestamp(info["created_at"]).strftime("%Y-%m-%d %H:%M:%S"),
            }
            for tid, info in ordered
        ],
    }


@app.get("/health", summary="ตรวจสอบสถานะ API")
async def health():
    return {"status": "ok", "model_loaded": "llm" in MODEL_STATE, "queue_size": queue.qsize()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("VLLM_API:app", host=SERVER_HOST, port=SERVER_PORT, reload=False)
