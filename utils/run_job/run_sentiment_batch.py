import time
from typing import Optional
from vllm import SamplingParams
from utils.sentiment_funtion import generate_sentiment_prompt, clean_sentiment


def run_sentiment_batch(
    self,
    ids: list,
    texts: list[str],
    max_tokens: Optional[int],
    temperature: Optional[float],
    prompt_file: str = "default.txt",
) -> dict:
    tokenizer = self.model_state["tokenizer"]
    prompts = []
    prompt_texts = []
    lengthed_texts = []
    for text in texts:
        prompt_text, lengthed_text = generate_sentiment_prompt(text, prompt_file)
        prompt_texts.append(prompt_text)
        lengthed_texts.append(lengthed_text)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}], tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)
    params = SamplingParams(
        temperature=temperature if temperature is not None else self.DEFAULT_TEMPERATURE,
        max_tokens=3,
        seed=self.SEED,
    )
    start = time.time()
    outputs = self.model_state["llm"].generate(prompts, params)
    stop = time.time()
    results = []
    for i, o in enumerate(outputs):
        raw = o.outputs[0].text.strip()
        cleaned_result = clean_sentiment(raw)
        results.append({
            "id": ids[i],
            "raw_text": texts[i],
            "lengthed_text": lengthed_texts[i],
            "final_prompt": prompt_texts[i],
            "sentiment": cleaned_result,
        })
    return {
        "time_usage_ms": (stop - start) * 1000,
        "token_usage": {
            "prompt_bytes": sum(len(p.encode()) for p in prompt_texts),
            "prompt_tokens": sum(len(o.prompt_token_ids) for o in outputs),
            "completion_tokens": sum(len(o.outputs[0].token_ids) for o in outputs),
            "total_tokens": sum(len(o.prompt_token_ids) + len(o.outputs[0].token_ids) for o in outputs),
        },
        "result": results,
    }
