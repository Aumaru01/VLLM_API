import re
from collections import Counter

WORDCLOUD_MAX_ROWS = 1000
WORDCLOUD_DEFAULT_MAX_TOKENS = 1024

# Structured-output schema the model must follow for each text
WORDCLOUD_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "words": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "word": {"type": "string"},
                    "count": {"type": "integer", "minimum": 1},
                },
                "required": ["word", "count"],
            },
        }
    },
    "required": ["words"],
}


def generate_wordcloud_prompt(text: str) -> str:
    return (
        "คุณเป็นผู้เชี่ยวชาญด้าน Natural Language Processing หน้าที่ของคุณคือการสกัดคำสำคัญเพื่อนำไปทำ Word Cloud\n"
        "คำสั่ง:\n"
        "- สกัดคำหรือวลีสำคัญ (keyword) ที่สื่อความหมายของข้อความ เช่น ชื่อเฉพาะ คำนาม หัวข้อ ประเด็น แบรนด์ สินค้า\n"
        "- ไม่ต้องใส่คำเชื่อม คำบุพบท คำสรรพนาม คำลงท้าย สัญลักษณ์ ตัวเลขลอย ๆ หรือ URL\n"
        "- ใช้คำตามที่ปรากฏในข้อความ ไม่แปลภาษา\n"
        "- count คือจำนวนครั้งที่คำนั้นปรากฏในข้อความ\n"
        "- ตอบเป็น JSON รูปแบบ {\"words\": [{\"word\": \"คำ\", \"count\": จำนวน}, ...]} เท่านั้น\n"
        f"ข้อความที่ต้องการวิเคราะห์: '{text}'"
    )


def _normalize_word(word: str) -> str:
    return re.sub(r"\s+", " ", str(word)).strip().lower()


def aggregate_wordcloud(per_text_words: list[list[dict]]) -> list[dict]:
    """Sum counts of the same word across all texts, sorted by count desc."""
    counter: Counter = Counter()
    display: dict[str, str] = {}
    for words in per_text_words:
        for item in words:
            if not isinstance(item, dict):
                continue
            key = _normalize_word(item.get("word", ""))
            if not key:
                continue
            try:
                count = max(int(item.get("count", 1)), 1)
            except (TypeError, ValueError):
                count = 1
            counter[key] += count
            display.setdefault(key, str(item["word"]).strip())
    return [{"word": display[k], "count": c} for k, c in counter.most_common()]
