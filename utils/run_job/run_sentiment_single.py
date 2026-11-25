import time
from typing import Optional
from vllm import SamplingParams
from utils.sentiment_funtion import generate_sentiment_prompt, clean_sentiment


def run_sentiment_single(
    self,
    text: str,
    max_tokens: Optional[int],
    temperature: Optional[float],
    prompt_file: str = "default.txt",
) -> dict:
    tokenizer = self.model_state["tokenizer"]
    prompt_text, lengthed_text = generate_sentiment_prompt(text, prompt_file)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}], tokenize=False, add_generation_prompt=True
    )
    params = SamplingParams(
        temperature=temperature if temperature is not None else self.DEFAULT_TEMPERATURE,
        max_tokens=3,
        seed=self.SEED,
    )
    start = time.time()
    outputs = self.model_state["llm"].generate([prompt], params)
    stop = time.time()
    raw = outputs[0].outputs[0].text.strip()
    cleaned_result = clean_sentiment(raw)
    return {
        "time_usage_ms": (stop - start) * 1000,
        "token_usage": {
            "prompt_bytes": len(prompt_text.encode()),
            "prompt_tokens": len(outputs[0].prompt_token_ids),
            "completion_tokens": len(outputs[0].outputs[0].token_ids),
            "total_tokens": len(outputs[0].prompt_token_ids) + len(outputs[0].outputs[0].token_ids),
        },
        "result": {
            "raw_text": text,
            "lengthed_text": lengthed_text,
            "final_prompt": prompt_text,
            "sentiment": cleaned_result,
        },
    }
