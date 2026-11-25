from typing import Optional
from fastapi import APIRouter, HTTPException

from __init__ import DEFAULT_TEMPERATURE
from utils.state import MODEL_STATE, _enqueue
from utils.schema import UniTextItem

router = APIRouter()


@router.post("/general", summary="Queue generation of general text from a single chat text", status_code=202)
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
