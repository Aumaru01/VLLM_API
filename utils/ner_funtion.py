import json
import re

from __init__ import SENTENCE_LENGTH_LIMIT, LENGTH_LIMIT, NER_TAG

def generate_ner_prompt(text):
    if SENTENCE_LENGTH_LIMIT:
        lengthed_text = text[:LENGTH_LIMIT]
    else:
        lengthed_text = text
    PROMPT = (
            f"คุณเป็นผู้เชี่ยวชาญด้านการทำ Natural Language Processing หน้าที่ของคุณคือการทำ Name entity recognition จากข้อความ"
            f"ข้อความที่ต้องการวิเคราะห์ได้แก่: '{lengthed_text}' "
            f"คำสั่ง: ทำ Name entity recognition โดยตอบในรูปแบบ JSON markdown ที่มี Key เป็น NER Tag และ Value เป็น list ของคำ เช่น"
            f"NER Tag1: [word1, word2, word3, ...], NER Tag2: [word1, word2, word3, ...]"
            f"โดยกำหนด NER Tag ดังนี้ {NER_TAG}"
        )
    return PROMPT, lengthed_text