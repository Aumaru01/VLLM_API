import time
from datetime import datetime
from fastapi import APIRouter

from utils.state import task_store, queue
from utils.funtion import _scan_disk_tasks

router = APIRouter()


@router.get("/queue", summary="List task ids and status from the current run plus disk history")
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
