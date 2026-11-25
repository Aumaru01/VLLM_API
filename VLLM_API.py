import os
import time
import asyncio
import contextlib

from vllm import LLM
from pathlib import Path
from transformers import AutoTokenizer
from contextlib import asynccontextmanager
from fastapi import FastAPI

from __init__ import (
    CFG,
    SERVER_HOST,
    SERVER_PORT,
    MODEL_PATH,
    QUANTIZATION,
    MODEL_DTYPE,
    MODEL_GPU_UTIL,
    MODEL_TRUST_REMOTE,
    ENFORCE_EAGER,
    RESULT_RETENTION_DAYS
)
from utils.run_job import JobRunner
from utils.funtion import _save_result
from utils.logger import logger
from utils.state import MODEL_STATE, task_store, queue
from router import ALL_ROUTERS

os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

# systemctl start VLLM_API.service
# systemctl status VLLM_API.service
# systemctl restart VLLM_API.service
# systemctl stop VLLM_API.service

# ============================================================
# Inference jobs (run off the event loop via asyncio.to_thread)
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    
    logger.info(f"[Startup] Loading vLLM model: {MODEL_PATH}")
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
    MODEL_STATE["run_text_with_image_single"] = JOBRUNNER.run_text_with_image_single
    MODEL_STATE["run_sentiment_single"] = JOBRUNNER.run_sentiment_single
    MODEL_STATE["run_sentiment_batch"] = JOBRUNNER.run_sentiment_batch
    MODEL_STATE["run_ner_single"] = JOBRUNNER.run_ner_single
    MODEL_STATE["run_ner_batch"] = JOBRUNNER.run_ner_batch
    MODEL_STATE["run_VTT_summary_single"] = JOBRUNNER.run_VTT_summary_single
    
    logger.info("[Startup] Model loaded successfully.")
    worker_task = asyncio.create_task(_worker())

    yield
    logger.info("[Shutdown] Stopping worker and releasing model state.")
    worker_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await worker_task
    MODEL_STATE.clear()

def _check_expired_files(cutoff_days: int = RESULT_RETENTION_DAYS) -> None:
    try:
        RESULT_DIR = Path("./result")
        cutoff_time = time.time() - (cutoff_days * 86400)
    
        for file_path in RESULT_DIR.rglob("*"):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()  # Deletes file in pathlib
                    logger.info(f"Deleted expired result file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting {file_path}: {e}")
    except Exception as e:
        logger.error(f"Directory processing error: {e}")


async def _worker() -> None:
    while True:
        job = await queue.get()
        task_id = job["task_id"]
        task_store[task_id]["status"] = "running"
        task_type = job["task_type"]
        logger.info(f"[Job start] task_id={task_id} task_type={task_type}")
        try:
            _check_expired_files()  # Check and delete expired files before processing the job

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
            
            elif task_type == "text_with_image_single":
                handler_result = await asyncio.to_thread(
                    MODEL_STATE["run_text_with_image_single"],
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
            logger.info(f"[Job done] task_id={task_id} task_type={task_type} time_used_ms={time_used_ms}")
        except Exception as e:
            task_store[task_id] = {**task_store[task_id], "status": "error", "error": str(e)}
            logger.exception(f"[Job error] task_id={task_id} task_type={task_type}: {e}")
        _save_result(task_id, task_store[task_id])
        queue.task_done()


app = FastAPI(
    title="vLLM Inference API",
    description=f"Generic LLM inference API — accepts chat messages, returns generated text\n\n model path: {MODEL_PATH}",
    lifespan=lifespan,
)

# ============================================================
# Endpoints (one file per endpoint in router/)
# ============================================================
for _router in ALL_ROUTERS:
    app.include_router(_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("VLLM_API:app", host=SERVER_HOST, port=SERVER_PORT, reload=False)
