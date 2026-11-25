import time
import pandas as pd
from io import BytesIO
from pathlib import Path
from typing import Optional
from vllm import SamplingParams
from utils.funtion import _parse_json_markdown, clean_text
from __init__ import VTT_SUMMARY_REQUIRED_COLUMNS, VTT_SUMMARY_SYSTEM_INSTRUCTION_DIR


def run_VTT_summary_single(
    self,
    file_bytes: bytes,
    max_tokens: Optional[int],
    temperature: Optional[float],
    system_instruction: str = "system_instruction",
) -> dict:
    tokenizer = self.model_state["tokenizer"]

    df = pd.read_excel(BytesIO(file_bytes))
    missing_columns = [col for col in VTT_SUMMARY_REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    df = df[VTT_SUMMARY_REQUIRED_COLUMNS].dropna(subset=["Text"])
    df["Text"] = df["Text"].map(clean_text)
    data_json = df.to_json(orient="records", force_ascii=False)

    system_instruction_filename = Path(system_instruction).name
    if ".txt" in system_instruction_filename:
        system_instruction_path = VTT_SUMMARY_SYSTEM_INSTRUCTION_DIR / f"{system_instruction_filename}"
    else:
        system_instruction_path = VTT_SUMMARY_SYSTEM_INSTRUCTION_DIR / f"{system_instruction_filename}.txt"
    if not system_instruction_path.is_file():
        raise ValueError(f"System instruction file not found: {system_instruction_path}")
    system_instruction_text = system_instruction_path.read_text(encoding="utf-8")
    prompt_text = f"{system_instruction_text}\n{data_json}"

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
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
    prompt_tokens = len(outputs[0].prompt_token_ids)
    completion_tokens = len(outputs[0].outputs[0].token_ids)
    return {
        "result": _parse_json_markdown(raw),
        "time_usage_ms": (stop - start) * 1000,
        "token_usage": {
            "prompt_bytes": len(prompt_text.encode()),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
