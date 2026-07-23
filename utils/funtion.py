import re
import json
import time
import hashlib
import unicodedata

from __init__ import GENERAL_DIR, SENTIMENT_DIR, RESULT_DIR, NER_DIR


def _save_result(task_id: str, data: dict) -> None:
    if "general" in task_id:
        (GENERAL_DIR / f"{task_id}.json").write_text(json.dumps(data, ensure_ascii=False))
    elif "sentiment" in task_id:
        (SENTIMENT_DIR / f"{task_id}.json").write_text(json.dumps(data, ensure_ascii=False))
    elif "ner" in task_id:
        (NER_DIR / f"{task_id}.json").write_text(json.dumps(data, ensure_ascii=False))
    else:
        (RESULT_DIR / f"{task_id}.json").write_text(json.dumps(data, ensure_ascii=False))

def _load_result(task_id: str) -> dict | None:
    if "general" in task_id:
        path = GENERAL_DIR / f"{task_id}.json"
    elif "sentiment" in task_id:
        path = SENTIMENT_DIR / f"{task_id}.json"
    elif "ner" in task_id:
        path = NER_DIR / f"{task_id}.json"
    else:
        path = RESULT_DIR / f"{task_id}.json"

    if path.exists():
        return json.loads(path.read_text())
    else:
        path = RESULT_DIR / f"{task_id}.json"
        if path.exists():
            return json.loads(path.read_text())
    return None

def _make_id(prefix: str) -> str:
    raw = str(time.time_ns())
    return f"{prefix}_{hashlib.shake_128(raw.encode()).hexdigest(8)}"

def _scan_disk_tasks(cutoff: float) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for directory in (GENERAL_DIR, SENTIMENT_DIR, NER_DIR, RESULT_DIR):
        for path in directory.glob("*.json"):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime < cutoff:
                continue
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            created_at = data.get("created_at", mtime)
            if created_at < cutoff:
                continue
            found[path.stem] = {"status": data.get("status"), "created_at": created_at}
    return found


def _parse_json_markdown(text: str):
    text = text.strip()

    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, text, re.IGNORECASE)

    # ถ้าเจอ Markdown ให้ใช้ข้อความข้างใน ถ้าไม่เจอให้ใช้ text เดิมเลย
    json_str = match.group(1).strip() if match else text

    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, Exception):
        # คืนค่า dict เปล่าทันทีถ้าแปลงไม่ผ่าน
        return {}

def clean_text(
    text: str, 
    remove_urls: bool = True, 
    remove_html: bool = True,
    remove_dots: bool = True,      # ลบจุดไข่ปลา / จุดซ้ำ
    remove_emojis: bool = False
) -> str:
    """
    ทำความสะอาดข้อความ (Clean Text) แบบครอบคลุม
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. แปลงอักขระให้อยู่ในรูปมาตรฐาน (Normalize Unicode)
    text = unicodedata.normalize('NFKC', text)

    # 2. ลบอักขระล่องหน (Zero-width characters)
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)

    # 3. ลบ HTML Tags
    if remove_html:
        text = re.sub(r'<[^>]+>', ' ', text)

    # 4. ลบ URL และ Email
    if remove_urls:
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+\.\S+', '', text)

    # 5. ลบจุดไข่ปลา (...) หรือ สัญลักษณ์จุดสี่ตัวขึ้นไป (… / . . . / ...)
    if remove_dots:
        # ลบสัญลักษณ์จุดไข่ปลาแบบ Unicode (… / Unicode \u2026)
        text = text.replace('…', ' ')
        # ลบจุดที่เรียงกันตั้งแต่ 2 จุดขึ้นไป (เช่น .. หรือ ...)
        text = re.sub(r'\.{2,}', ' ', text)

    # 6. ลบ Emoji (ถ้าต้องการ)
    if remove_emojis:
        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)

    # 7. ยุบช่องว่าง/เว้นวรรค/ขึ้นบรรทัดใหม่ ซ้ำๆ ให้เหลือช่องว่างเดียว
    text = re.sub(r'\s+', ' ', text)

    # 8. ลบ Space หัว-ท้ายข้อความ
    return text.strip()