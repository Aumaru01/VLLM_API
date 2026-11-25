import time
import json
from typing import Optional
from vllm import SamplingParams
from pythainlp.tokenize import sent_tokenize
from collections import defaultdict, Counter
from utils.ner_funtion import generate_ner_prompt
from utils.funtion import _parse_json_markdown, clean_text
from __init__ import NER_MINIMUM_COUNT


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

    output = {}
    # วน loop ลุยไปทีละ key และ list ของ values
    for key, values in output_entities.items():

        # นับจำนวน items ที่ซ้ำกันใน values
        counts = Counter(values)

        # คัดเลือกเฉพาะตัวที่นับได้มากกว่า NER_MINIMUM_COUNT
        filtered_list = []
        for value, count in counts.items():
            if count > NER_MINIMUM_COUNT:
                filtered_list.append({value: count})

        # เก็บผลลัพธ์ใส่ output
        output[key] = filtered_list

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
