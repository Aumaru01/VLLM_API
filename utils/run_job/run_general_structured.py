import time
import json
from typing import Optional
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams


def run_general_structured(
    self,
    text: str, 
    json_schema: dict, 
    max_tokens: Optional[int], 
    temperature: Optional[float]
) -> dict:
    tokenizer = self.model_state["tokenizer"]
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )
    params = SamplingParams(
        temperature=temperature if temperature is not None else self.DEFAULT_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else self.DEFAULT_MAX_TOKENS,
        structured_outputs=StructuredOutputsParams(json=json_schema),
        seed=self.SEED,
    )
    start = time.time()
    outputs = self.model_state["llm"].generate([prompt], params)
    stop = time.time()
    raw = outputs[0].outputs[0].text.strip()
    prompt_tokens = len(outputs[0].prompt_token_ids)
    completion_tokens = len(outputs[0].outputs[0].token_ids)
    token_usage = {
        "prompt_bytes": len(text.encode()),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = raw
    return {
        "result": result,
        "time_usage_ms": (stop - start) * 1000,
        "token_usage": token_usage,
    }
