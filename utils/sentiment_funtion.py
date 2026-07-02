
import re
import yaml


with open("config.yaml") as f:
    _cfg = yaml.safe_load(f)

_SENTIMENT_CFG = _cfg.get("sentiment", {})
LENGTH_LIMIT: int = _SENTIMENT_CFG.get("max_string_length", 500)
FILTER: bool = _SENTIMENT_CFG.get("only_sentiment_output", True)


def generate_sentiment_prompt(text):
    lengthed_text = text[:LENGTH_LIMIT]
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