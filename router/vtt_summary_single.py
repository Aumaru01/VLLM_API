from typing import Optional
from fastapi import APIRouter, HTTPException, File, Form, UploadFile

from __init__ import DEFAULT_TEMPERATURE
from utils.state import MODEL_STATE, _enqueue

router = APIRouter()


@router.post("/VTT_summary_single", summary="Queue generation of VTT summary from an uploaded Excel file", status_code=202)
async def VTT_summary_single(
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
