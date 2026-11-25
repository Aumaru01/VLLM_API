from typing import Optional
from fastapi import APIRouter, HTTPException, File, Form, UploadFile

from __init__ import DEFAULT_TEMPERATURE
from utils.state import MODEL_STATE, _enqueue

router = APIRouter()


@router.post("/text_with_image_single", summary="Queue generation of general text from a chat text with an uploaded image file", status_code=202)
async def text_with_image_single(
    file: UploadFile = File(...),
    text: Optional[str] = Form(None),
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = DEFAULT_TEMPERATURE,
):
    if "llm" not in MODEL_STATE:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    is_image = (file.content_type or "").startswith("image/") or (file.filename or "").lower().endswith(
        (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
    )
    if not is_image:
        raise HTTPException(status_code=422, detail="Only image files are supported")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")

    task_id = _enqueue(
        "text_with_image_single",
        text=text,
        file_bytes=file_bytes,
        filename=file.filename,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return {"task_id": task_id, "status": "queued"}
