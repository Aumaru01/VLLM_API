from fastapi import APIRouter, HTTPException

from utils.state import task_store
from utils.funtion import _load_result

router = APIRouter()


@router.get("/result/{task_id}", summary="ดึงสถานะ/ผลลัพธ์ของ task จาก task_id")
async def get_result(task_id: str):
    task = task_store.get(task_id)
    if task is not None:
        return {"task_id": task_id, **task}

    saved = _load_result(task_id)
    if saved is not None:
        return {"task_id": task_id, **saved}

    raise HTTPException(status_code=404, detail="task_id not found")
