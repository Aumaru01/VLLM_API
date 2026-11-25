import time
import json
from typing import Optional
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams


def run_general_batch_structured(
    self,
    ids: list, 
    texts: list[str], 
    json_schema: dict, 
    max_tokens: Optional[int], 
    temperature: Optional[float]
) -> dict:
    tokenizer = self.model_state["tokenizer"]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True
        )
        for msg in texts
    ]
    params = SamplingParams(
        temperature=temperature if temperature is not None else self.DEFAULT_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else self.DEFAULT_MAX_TOKENS,
        structured_outputs=StructuredOutputsParams(json=json_schema),
        seed=self.SEED,
    )
    start = time.time()
    outputs = self.model_state["llm"].generate(prompts, params)
    stop = time.time()
    results = []
    for i, o in enumerate(outputs):
        raw = o.outputs[0].text.strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        results.append({"id": ids[i], "result": parsed})
    prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    completion_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    return {
        "result": results,
        "time_usage_ms": (stop - start) * 1000,
        "token_usage": {
            "prompt_bytes": sum(len(p.encode()) for p in texts),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
