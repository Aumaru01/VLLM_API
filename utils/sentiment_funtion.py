
import re
import yaml
from __init__ import SENTENCE_LENGTH_LIMIT, LENGTH_LIMIT, FILTER


def generate_sentiment_prompt(text):
    if SENTENCE_LENGTH_LIMIT:
        lengthed_text = text[:LENGTH_LIMIT]
    else:
        lengthed_text = text
    PROMPT = (
            # f"คุณเป็นผู้เชี่ยวชาญด้าน Sentiment Analysis หน้าที่ของคุณคือการประเมิน 'ความรู้สึกของผู้คอมเมนต์' เท่านั้น "
            f"คุณเป็นผู้เชี่ยวชาญด้าน Sentiment Analysis ที่มักจะมองโลกในแง่ร้าย หน้าที่ของคุณคือการประเมิน 'ความรู้สึกของผู้คอมเมนต์' เท่านั้น "
            f"คอมเมนต์ที่ต้องการวิเคราะห์: '{lengthed_text}' "
            f"คำสั่ง: จากเนื้อหาของต้นโพสต์ จงวิเคราะห์ความรู้สึกของผู้คอมเมนต์ว่ามีทิศทางใด "
            f"เลือกตอบเพียงคำเดียว (Positive, Neutral, หรือ Negative) ห้ามมีคำอธิบายเพิ่มเติม"
            # f"กรณีไม่มีคอมเมนต์ ให้ตอบเป็น Neutral"
        )
    return PROMPT, lengthed_text

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