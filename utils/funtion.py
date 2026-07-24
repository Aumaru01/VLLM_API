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

def remove_duplicate_phrases(text: str) -> str:
    """
    ฟังก์ชันช่วยจัดการข้อความส่วนหน้าที่มีการขาดคำแล้วซ้ำกับประโยคเต็มด้านหลัง
    เช่น 'ต้นทุนดําเนินธุรกิ . . . . SME ไทยกําลังเผชิญ...'
    """
    # 1. ลบจุดไข่ปลาแบบคั่นด้วย space (. . . .) ก่อนเพื่อให้ข้อความต่อกัน
    text = re.sub(r'(\.\s*){2,}', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 2. ตรวจหาคำสั้นๆ ที่ลอยอยู่แล้วไปซ้ำกับจุดเริ่มต้นของประโยคถัดไป
    words = text.split()
    for i in range(len(words) // 2, 0, -1):
        phrase = " ".join(words[:i])
        # ตัดคำสุดท้ายของวลีออก (เช่น ตัดคำว่า 'ธุรกิ' ออก) เพื่อดูว่าส่วนที่เหลือไปตรงกับประโยคข้างหลังไหม
        phrase_stem = " ".join(words[:i-1]) if i > 1 else ""
        rest_of_text = " ".join(words[i:])
        
        if phrase_stem and phrase_stem in rest_of_text:
            return rest_of_text
            
    return text

def clean_text(
    text: str, 
    remove_urls: bool = True, 
    remove_html: bool = True,
    remove_dots: bool = True,
    deduplicate: bool = True
) -> str:
    if not text or not isinstance(text, str):
        return ""

    # 1. Normalize Unicode & ลบอักขระล่องหน
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)

    # 2. ลบ HTML / URLs
    if remove_html:
        text = re.sub(r'<[^>]+>', ' ', text)
    if remove_urls:
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # 3. จัดการข้อความซ้ำซ้อนจากจุดไข่ปลา (ตัดส่วนซ้ำด้านหน้าออก)
    if deduplicate:
        text = remove_duplicate_phrases(text)

    # 4. ลบจุดไข่ปลา / จุดซ้ำทุกรูปแบบ
    if remove_dots:
        text = text.replace('…', ' ')
        text = re.sub(r'(\.\s*){2,}', ' ', text) # ลบทั้ง .... และ . . . .
        text = re.sub(r'\.{2,}', ' ', text)

    # 5. ยุบ Space ซ้ำ และตัด Space หัว-ท้าย
    text = re.sub(r'\s+', ' ', text)
    return text.strip()