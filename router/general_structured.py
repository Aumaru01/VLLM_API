from typing import Optional
from fastapi import APIRouter, HTTPException

from __init__ import DEFAULT_TEMPERATURE
from utils.state import MODEL_STATE, _enqueue
from utils.schema import GenerateStructuredRequest

router = APIRouter()


@router.post("/general_structured", summary="Queue generation of general structured JSON output from a single text", status_code=202)
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
