import json
import re

from __init__ import SENTENCE_LENGTH_LIMIT, LENGTH_LIMIT, NER_TAG

def generate_ner_prompt(text):
    if SENTENCE_LENGTH_LIMIT:
        lengthed_text = text[:LENGTH_LIMIT]
    else:
        lengthed_text = text
    PROMPT = (
            # f"คุณเป็นผู้เชี่ยวชาญด้าน Sentiment Analysis หน้าที่ของคุณคือการประเมิน 'ความรู้สึกของผู้คอมเมนต์' เท่านั้น "
            f"คุณเป็นผู้เชี่ยวชาญด้านการทำ Natural Language Processing หน้าที่ของคุณคือการทำ Name entity recognition จากคอมเมนต์"
            f"คอมเมนต์ที่ต้องการวิเคราะห์: '{lengthed_text}' "
            f"คำสั่ง: ทำ Name entity recognition โดยตอบในรูปแบบ JSON markdown ที่มี Key เป็น NER Tag และ Value เป็น list ของคำ"
            f"โดยกำหนด NER Tag ดังนี้ {NER_TAG}"
            # f"กรณีไม่มีคอมเมนต์ ให้ตอบเป็น Neutral"
        )
    return PROMPT, lengthed_text