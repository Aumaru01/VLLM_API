import time
import asyncio

from utils.funtion import _make_id
from utils.logger import logger

# ============================================================
# Global state (shared between VLLM_API.py and router/*)
# ============================================================
MODEL_STATE: dict = {}
task_store: dict[str, dict] = {}
queue: asyncio.Queue = asyncio.Queue()


def _enqueue(task_type: str, **job_fields) -> str:
    task_id = _make_id(task_type)
    task_store[task_id] = {"status": "queued", "created_at": time.time()}
    queue.put_nowait({"task_id": task_id, "task_type": task_type, **job_fields})
    logger.info(f"[Job queued] task_id={task_id} task_type={task_type} queue_size={queue.qsize()}")
    return task_id
