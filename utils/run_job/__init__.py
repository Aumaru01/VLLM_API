from utils.run_job.run_general_single import run_general_single
from utils.run_job.run_general_batch import run_general_batch
from utils.run_job.run_general_structured import run_general_structured
from utils.run_job.run_general_batch_structured import run_general_batch_structured
from utils.run_job.run_general_with_PDF import run_general_with_PDF
from utils.run_job.run_text_with_image_single import run_text_with_image_single
from utils.run_job.run_sentiment_single import run_sentiment_single
from utils.run_job.run_sentiment_batch import run_sentiment_batch
from utils.run_job.run_ner_single import run_ner_single
from utils.run_job.run_ner_batch import run_ner_batch
from utils.run_job.run_VTT_summary_single import run_VTT_summary_single
from utils.run_job.run_wordcloud import run_wordcloud


# ============================================================
# Funtions to run jobs (each method lives in its own file in utils/run_job/)
# ============================================================
class JobRunner:
    def __init__(self, config, model_state):
        _cfg = config
        self.DEFAULT_MAX_TOKENS: int = _cfg["inference"]["default_max_tokens"]
        self.DEFAULT_TEMPERATURE: float = _cfg["inference"]["default_temperature"]
        self.SEED: int = _cfg["inference"]["seed"]
        self.model_state = model_state

    run_general_single = run_general_single
    run_general_batch = run_general_batch
    run_general_structured = run_general_structured
    run_general_batch_structured = run_general_batch_structured
    run_general_with_PDF = run_general_with_PDF
    run_text_with_image_single = run_text_with_image_single
    run_sentiment_single = run_sentiment_single
    run_sentiment_batch = run_sentiment_batch
    run_ner_single = run_ner_single
    run_ner_batch = run_ner_batch
    run_VTT_summary_single = run_VTT_summary_single
    run_wordcloud = run_wordcloud
