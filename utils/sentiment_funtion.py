
import re
import yaml
from pathlib import Path
from __init__ import SENTENCE_LENGTH_LIMIT, LENGTH_LIMIT, FILTER, SENTIMENT_PROMPT_DIR

def _load_sentiment_prompt(prompt_file: str) -> str:
    prompt_filename = Path(prompt_file).name
    if prompt_filename.endswith(".txt"):
        prompt_path = SENTIMENT_PROMPT_DIR / prompt_filename
    else:
        prompt_path = SENTIMENT_PROMPT_DIR / f"{prompt_filename}.txt"
    if not prompt_path.is_file():
        raise ValueError(f"Sentiment prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")

def generate_sentiment_prompt(text, prompt_file: str = "default.txt"):
    if SENTENCE_LENGTH_LIMIT:
        lengthed_text = text[:LENGTH_LIMIT]
    else:
        lengthed_text = text

    prompt = _load_sentiment_prompt(prompt_file)
    input_prompt = prompt+f"คอมเมนต์ที่ต้องการวิเคราะห์: '{lengthed_text}'"

    return input_prompt, lengthed_text

def clean_sentiment(sentiment):
    sentiment = re.sub(r"\s+", "", sentiment)
    if FILTER == True:
        allowed_sentiments = {'Positive', 'Neutral', 'Negative'}
        lower_allowed_sentiments = {'positive', 'neutral', 'negative'}
        if sentiment in allowed_sentiments:
            return sentiment
        if sentiment in lower_allowed_sentiments:
            return sentiment.capitalize()
        return 'Neutral'
    else:
        return sentiment