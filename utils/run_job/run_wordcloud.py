import time
import json
from typing import Optional
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from utils.wordcloud_funtion import (
    WORDCLOUD_JSON_SCHEMA,
    WORDCLOUD_DEFAULT_MAX_TOKENS,
    generate_wordcloud_prompt,
    aggregate_wordcloud,
    clean_wordcloud_text,
)


def run_wordcloud(
    self,
    ids: list,
    texts: list[Optional[str]],
    input_rows: int,
    max_tokens: Optional[int],
    temperature: Optional[float],
) -> dict:
    tokenizer = self.model_state["tokenizer"]

    # Clean (HTML, URL, emoji, ...) then skip texts that end up empty
    valid = [(i, clean_wordcloud_text(t)) for i, t in enumerate(texts)]
    valid = [(i, t) for i, t in valid if t]

    prompt_texts = [generate_wordcloud_prompt(t) for _, t in valid]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompt_texts
    ]
    params = SamplingParams(
        temperature=temperature if temperature is not None else self.DEFAULT_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else WORDCLOUD_DEFAULT_MAX_TOKENS,
        structured_outputs=StructuredOutputsParams(json=WORDCLOUD_JSON_SCHEMA),
        seed=self.SEED,
    )
    start = time.time()
    outputs = self.model_state["llm"].generate(prompts, params) if prompts else []
    stop = time.time()

    per_id_words: dict[int, list] = {i: [] for i in range(len(ids))}
    failed_ids = []
    for (idx, _), o in zip(valid, outputs):
        raw = o.outputs[0].text.strip()
        try:
            parsed = json.loads(raw)
            words = parsed.get("words", []) if isinstance(parsed, dict) else []
        except json.JSONDecodeError:
            words = []
            failed_ids.append(ids[idx])
        per_id_words[idx] = words

    words = aggregate_wordcloud(list(per_id_words.values()))
    prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    completion_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    return {
        "time_usage_ms": (stop - start) * 1000,
        "token_usage": {
            "prompt_bytes": sum(len(p.encode()) for p in prompt_texts),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "result": {
            "input_rows": input_rows,
            "used_rows": len(ids),
            "trimmed": input_rows > len(ids),
            "unique_words": len(words),
            "words": words,
            "per_id": [{"id": ids[i], "words": per_id_words[i]} for i in range(len(ids))],
            "failed_ids": failed_ids,
        },
    }
