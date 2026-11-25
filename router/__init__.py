from router import (
    general,
    general_batch,
    general_structured,
    general_batch_structured,
    general_with_pdf,
    text_with_image_single,
    sentiment,
    sentiment_batch,
    ner,
    ner_batch,
    vtt_summary_single,
    result,
    task_queue,
    health,
)

# Order here = order shown in /docs
ALL_ROUTERS = [
    general.router,
    general_batch.router,
    general_structured.router,
    general_batch_structured.router,
    general_with_pdf.router,
    text_with_image_single.router,
    sentiment.router,
    sentiment_batch.router,
    ner.router,
    ner_batch.router,
    vtt_summary_single.router,
    result.router,
    task_queue.router,
    health.router,
]
