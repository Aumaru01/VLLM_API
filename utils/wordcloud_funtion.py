import re
import html
import unicodedata
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


# ============================================================
# Text cleaning (run on every text before it is sent to the LLM)
# ============================================================
_RE_SCRIPT_STYLE = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_URL = re.compile(
    r"(?:https?://|ftp://|www\.)\S+"                                  # http://..., www....
    r"|\b[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:com|net|org|co|th|io|me|ly|gl|info|biz|tv|app)\b(?:/\S*)?",  # bit.ly/xx, abc.co.th
    re.IGNORECASE,
)
_RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_RE_MENTION = re.compile(r"(?<!\w)@[^\s@]+")
_RE_HASHTAG = re.compile(r"#([^\s#]+)")   # [^\s#] not \w: \w misses Thai tone marks
_RE_INVISIBLE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060-\u2064\uFEFF\u00AD]")
_RE_EMOJI = re.compile(
    "["
    "\U0001F000-\U0001FAFF"   # emoji, pictographs, flags, symbols
    "\U00002600-\U000027BF"   # misc symbols & dingbats
    "\U00002B00-\U00002BFF"   # arrows, stars
    "\U0000FE00-\U0000FE0F"   # variation selectors
    "\U0001F1E6-\U0001F1FF"   # regional indicators
    "\u200D\u20E3"             # ZWJ, keycap
    "]+"
)
_RE_ELONGATED = re.compile(r"([^\d\s])\1{2,}")              # มากกกกก -> มาก, !!!!! -> !
_RE_DOTS = re.compile(r"(?:\.\s*){2,}|…")
_RE_PUNCT_RUN = re.compile(r"([!?,;:~\-_=*+|/\\^])\1+")
_RE_SPACES = re.compile(r"\s+")


def clean_wordcloud_text(text) -> str:
    """Clean a raw (often social-media / scraped) text before word cloud extraction.

    Removes HTML (tags, entities, script/style), URLs, emails, @mentions, emoji,
    invisible/control characters, elongated characters and punctuation runs.
    Hashtags keep their word but lose the '#'. Returns "" for empty / non-str input.
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. HTML: decode entities (twice for double-encoded &amp;lt;), drop script/style, drop tags
    text = html.unescape(html.unescape(text))
    text = _RE_SCRIPT_STYLE.sub(" ", text)
    text = _RE_HTML_TAG.sub(" ", text)

    # 2. Unicode normalize + remove invisible / control characters
    text = unicodedata.normalize("NFKC", text)
    text = _RE_INVISIBLE.sub("", text)
    text = "".join(ch if unicodedata.category(ch)[0] != "C" else " " for ch in text)

    # 3. Emails, URLs, mentions (emails first so "a@shop.co.th" is removed whole)
    text = _RE_EMAIL.sub(" ", text)
    text = _RE_URL.sub(" ", text)
    text = _RE_MENTION.sub(" ", text)

    # 4. Hashtags: #ข่าวเด่น -> ข่าวเด่น, #big_sale -> big sale
    text = _RE_HASHTAG.sub(lambda m: " " + m.group(1).replace("_", " ") + " ", text)

    # 5. Emoji / pictographs
    text = _RE_EMOJI.sub(" ", text)

    # 6. Ellipsis (before elongation so "..." isn't collapsed to "."), elongated chars, punctuation runs
    text = _RE_DOTS.sub(" ", text)
    text = _RE_ELONGATED.sub(r"\1", text)
    text = _RE_PUNCT_RUN.sub(r"\1", text)

    # 7. Collapse whitespace
    return _RE_SPACES.sub(" ", text).strip()


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
