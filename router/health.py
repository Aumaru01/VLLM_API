from fastapi import APIRouter

from utils.state import MODEL_STATE, queue

router = APIRouter()


@router.get("/health", summary="ตรวจสอบสถานะ API")
async def health():
    return {"status": "ok", "model_loaded": "llm" in MODEL_STATE, "queue_size": queue.qsize()}
