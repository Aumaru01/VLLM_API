from typing import Optional
from fastapi import APIRouter, HTTPException

from __init__ import DEFAULT_TEMPERATURE
from utils.state import MODEL_STATE, _enqueue
from utils.schema import MultiTextItem
from utils.wordcloud_funtion import WORDCLOUD_MAX_ROWS

router = APIRouter()


@router.post(
    "/wordcloud",
    summary=f"Queue word cloud extraction (word + count) from up to {WORDCLOUD_MAX_ROWS} texts",
    status_code=202,
)
async def wordcloud(
    req: list[MultiTextItem],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = DEFAULT_TEMPERATURE,
):
    if "llm" not in MODEL_STATE:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not req:
        raise HTTPException(status_code=422, detail="texts must not be empty")

    input_rows = len(req)
    req = req[:WORDCLOUD_MAX_ROWS]  # keep only the first (top) 1000 rows

    ids = [item.id for item in req]
    texts = [item.text for item in req]

    if len(set(ids)) != len(ids):
        raise HTTPException(status_code=422, detail="Duplicate ids found in texts")

    task_id = _enqueue(
        "wordcloud",
        ids=ids,
        texts=texts,
        input_rows=input_rows,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return {
        "task_id": task_id,
        "status": "queued",
        "input_rows": input_rows,
        "used_rows": len(req),
        "trimmed": input_rows > WORDCLOUD_MAX_ROWS,
    }
