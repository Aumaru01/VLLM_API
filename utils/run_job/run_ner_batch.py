import time
from typing import Optional


def run_ner_batch(
    self,
    ids: list,
    texts: list[str],
    max_tokens: Optional[int],
    temperature: Optional[float]
) -> dict:
    start = time.time()
    single_results = [
        self.run_ner_single(text, max_tokens, temperature) for text in texts
    ]
    stop = time.time()

    results = [
        {"id": ids[i], **single_results[i]["result"]}
        for i in range(len(texts))
    ]

    return {
        "time_usage_ms": (stop - start) * 1000,
        "token_usage": {
            "prompt_bytes": sum(r["token_usage"]["prompt_bytes"] for r in single_results),
            "prompt_tokens": sum(r["token_usage"]["prompt_tokens"] for r in single_results),
            "completion_tokens": sum(r["token_usage"]["completion_tokens"] for r in single_results),
            "total_tokens": sum(r["token_usage"]["total_tokens"] for r in single_results),
        },
        "result": results,
    }
