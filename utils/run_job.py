import re
import time
import yaml
import json
from pathlib import Path
import pypdfium2 as pdfium
from typing import Optional
from vllm import SamplingParams
from collections import defaultdict
from pythainlp.tokenize import sent_tokenize

from vllm.sampling_params import StructuredOutputsParams
from utils.sentiment_funtion import generate_sentiment_prompt, clean_sentiment
from utils.ner_funtion import generate_ner_prompt
from utils.funtion import _parse_json_markdown, clean_text


MAX_PDF_PAGES = 8

# ============================================================
# Funtions to run jobs
# ============================================================
class JobRunner:
    def __init__(self, config, model_state):
        _cfg = config
        self.DEFAULT_MAX_TOKENS: int = model_state["llm"].llm_engine.model_config.max_model_len
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

    def run_ner_single(
        self,
        text: str,
        max_tokens: Optional[int],
        temperature: Optional[float]
    ) -> dict:
        tokenizer = self.model_state["tokenizer"]
        params = SamplingParams(
            temperature=temperature if temperature is not None else self.DEFAULT_TEMPERATURE,
            max_tokens=max_tokens if max_tokens is not None else self.DEFAULT_MAX_TOKENS,
            seed=self.SEED,
        )
        text = clean_text(text)
        sentences = sent_tokenize(text, engine="crfcut")

        prompts = []
        prompt_texts = []
        lengthed_sentences = []
        for sentence in sentences:
            prompt_text, lengthed_text = generate_ner_prompt(sentence)
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}], tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)
            prompt_texts.append(prompt_text)
            lengthed_sentences.append(lengthed_text)

        output_entities = defaultdict(list)
        start = time.time()
        outputs = self.model_state["llm"].generate(prompts, params) if prompts else []
        stop = time.time()

        for o in outputs:
            raw = o.outputs[0].text.strip()
            json_output = _parse_json_markdown(raw)
            if isinstance(json_output, dict):
                for key, value in json_output.items():
                    if not isinstance(value, list):
                        continue
                    for item in value:
                        if isinstance(item, (dict, list)):
                            item = json.dumps(item, ensure_ascii=False)
                        output_entities[key].append(item)

        output = {
            key: [{value: values.count(value)} for value in dict.fromkeys(values)]
            for key, values in output_entities.items()
        }

        return {
            "time_usage_ms": (stop - start) * 1000,
            "token_usage": {
                "prompt_bytes": sum(len(p.encode()) for p in prompt_texts),
                "prompt_tokens": sum(len(o.prompt_token_ids) for o in outputs),
                "completion_tokens": sum(len(o.outputs[0].token_ids) for o in outputs),
                "total_tokens": sum(len(o.prompt_token_ids) + len(o.outputs[0].token_ids) for o in outputs),
            },
            "result": {
                "raw_text": text,
                "text": " ".join(lengthed_sentences),
                "output": output,
            },
        }

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