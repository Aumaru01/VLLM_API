import time
import yaml
import json
from pathlib import Path
from typing import Optional
from vllm import SamplingParams

from vllm.sampling_params import StructuredOutputsParams
from utils.sentiment_funtion import generate_sentiment_prompt, clean_sentiment


# ============================================================
# Funtions to run jobs
# ============================================================
class JobRunner:
    def __init__(self, config, model_state):
        _cfg = config
        self.DEFAULT_MAX_TOKENS: int = _cfg["inference"]["default_max_tokens"]
        self.DEFAULT_TEMPERATURE: float = _cfg["inference"]["default_temperature"]
        self.SEED: int = _cfg["inference"]["seed"]
        self.model_state = model_state

    def run_general_single(
        self,
        text: str, 
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
            seed=self.SEED,
        )
        start = time.time()
        outputs = self.model_state["llm"].generate([prompt], params)
        stop = time.time()
        prompt_tokens = len(outputs[0].prompt_token_ids)
        completion_tokens = len(outputs[0].outputs[0].token_ids)
        return {
            "text": outputs[0].outputs[0].text.strip(),
            "time_usage_ms": (stop - start) * 1000,
            "token_usage": {
                "prompt_bytes": len(text.encode()),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def run_general_batch(
        self,
        ids: list, 
        texts: list[str], 
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
            seed=self.SEED,
        )
        start = time.time()
        outputs = self.model_state["llm"].generate(prompts, params)
        stop = time.time()
        prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs)
        completion_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        return {
            "result": [
                {"id": ids[i], "text": o.outputs[0].text.strip()} for i, o in enumerate(outputs)
            ],
            "time_usage_ms": (stop - start) * 1000,
            "token_usage": {
                "prompt_bytes": sum(len(p.encode()) for p in texts),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

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

    def run_sentiment_single(
        self,
        text: str, 
        max_tokens: Optional[int], 
        temperature: Optional[float]
    ) -> dict:
        tokenizer = self.model_state["tokenizer"]
        prompt_text, lengthed_text = generate_sentiment_prompt(text)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}], tokenize=False, add_generation_prompt=True
        )
        params = SamplingParams(
            temperature=temperature if temperature is not None else self.DEFAULT_TEMPERATURE,
            max_tokens=max_tokens if max_tokens is not None else self.DEFAULT_MAX_TOKENS,
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
                "text": lengthed_text,
                "sentiment": cleaned_result,
            },
        }

    def run_sentiment_batch(
        self,
        ids: list, 
        texts: list[str], 
        max_tokens: Optional[int], 
        temperature: Optional[float]
    ) -> dict:
        tokenizer = self.model_state["tokenizer"]
        prompts = []
        prompt_texts = []
        for text in texts:
            prompt_text, lengthed_text = generate_sentiment_prompt(text)
            prompt_texts.append(prompt_text)
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}], tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)
        params = SamplingParams(
            temperature=temperature if temperature is not None else self.DEFAULT_TEMPERATURE,
            max_tokens=max_tokens if max_tokens is not None else self.DEFAULT_MAX_TOKENS,
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
                "text": prompt_texts[i],
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
