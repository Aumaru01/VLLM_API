import time
from io import BytesIO
from PIL import Image
from typing import Optional
from vllm import SamplingParams


def run_text_with_image_single(
    self,
    text: Optional[str],
    file_bytes: bytes,
    filename: str,
    max_tokens: Optional[int],
    temperature: Optional[float],
) -> dict:
    user_text = text or "Please describe the attached image."
    params = SamplingParams(
        temperature=temperature if temperature is not None else self.DEFAULT_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else self.DEFAULT_MAX_TOKENS,
        seed=self.SEED,
    )
    start = time.time()
    try:
        image = Image.open(BytesIO(file_bytes)).convert("RGB")

        content = [
            {"type": "text", "text": user_text},
            {"type": "image_pil", "image_pil": image},
        ]
        messages = [{"role": "user", "content": content}]

        outputs = self.model_state["llm"].chat(messages, sampling_params=params)
        stop = time.time()
        prompt_tokens = len(outputs[0].prompt_token_ids)
        completion_tokens = len(outputs[0].outputs[0].token_ids)
        return {
            "result": {
                "text": outputs[0].outputs[0].text.strip(),
                "note": None,
            },
            "time_usage_ms": (stop - start) * 1000,
            "token_usage": {
                "prompt_bytes": len(user_text.encode()) + len(file_bytes),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    except Exception as e:
        stop = time.time()
        return {
            "result": {
                "text": None,
                "note": "model not support upload image",
                "error": str(e),
                "filename": filename,
            },
            "time_usage_ms": (stop - start) * 1000,
            "token_usage": {
                "prompt_bytes": len(user_text.encode()),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
