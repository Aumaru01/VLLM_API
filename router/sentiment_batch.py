from typing import Optional
from fastapi import APIRouter, HTTPException

from __init__ import DEFAULT_TEMPERATURE
from utils.state import MODEL_STATE, _enqueue
from utils.schema import MultiTextItem

router = APIRouter()


@router.post("/sentiment_batch", summary="Queue sentiment analysis of multiple chat texts", status_code=202)
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
