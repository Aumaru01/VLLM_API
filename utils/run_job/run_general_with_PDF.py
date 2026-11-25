import time
import pypdfium2 as pdfium
from typing import Optional
from vllm import SamplingParams
from utils.run_job.constants import MAX_PDF_PAGES


def run_general_with_PDF(
    self,
    text: Optional[str],
    file_bytes: bytes,
    filename: str,
    max_tokens: Optional[int],
    temperature: Optional[float],
) -> dict:
    user_text = text or f"Please read the attached file: {filename}"
    params = SamplingParams(
        temperature=temperature if temperature is not None else self.DEFAULT_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else self.DEFAULT_MAX_TOKENS,
        seed=self.SEED,
    )
    start = time.time()
    try:
        pdf = pdfium.PdfDocument(file_bytes)
        page_count = len(pdf)
        pages_used = min(page_count, MAX_PDF_PAGES)
        page_images = [pdf[i].render(scale=2).to_pil() for i in range(pages_used)]

        content = [{"type": "text", "text": user_text}]
        content += [{"type": "image_pil", "image_pil": img} for img in page_images]
        messages = [{"role": "user", "content": content}]

        outputs = self.model_state["llm"].chat(messages, sampling_params=params)
        stop = time.time()
        prompt_tokens = len(outputs[0].prompt_token_ids)
        completion_tokens = len(outputs[0].outputs[0].token_ids)
        result = {
            "text": outputs[0].outputs[0].text.strip(),
            "note": None,
            "pages_used": pages_used,
            "pages_total": page_count,
        }
        if pages_used < page_count:
            result["note"] = f"only the first {pages_used} of {page_count} pages were processed"
        return {
            "result": result,
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
                "note": "model not support upload file",
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
