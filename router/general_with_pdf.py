from typing import Optional
from fastapi import APIRouter, HTTPException, File, Form, UploadFile

from __init__ import DEFAULT_TEMPERATURE
from utils.state import MODEL_STATE, _enqueue

router = APIRouter()


@router.post("/general_with_PDF", summary="Queue generation of general text from a chat text with an uploaded PDF file", status_code=202)
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
