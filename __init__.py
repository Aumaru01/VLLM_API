import yaml

from pathlib import Path

_cfg_path = Path(__file__).parent / "config.yaml"
with open(_cfg_path) as f:
    CFG = yaml.safe_load(f)

SERVER_HOST: str = CFG["server"]["host"]
SERVER_PORT: int = CFG["server"]["port"]

MODEL_PATH: str = CFG["model"]["path"]
QUANTIZATION: str = CFG["model"]["quantization"]
MODEL_DTYPE: str = CFG["model"]["dtype"]
MODEL_GPU_UTIL: float = CFG["model"]["gpu_memory_utilization"]
MODEL_TRUST_REMOTE: bool = CFG["model"]["trust_remote_code"]
ENFORCE_EAGER: bool = CFG["model"]["enforce_eager"]

DEFAULT_TEMPERATURE: float = CFG["inference"]["default_temperature"]
SEED: int = CFG["inference"]["seed"]

RESULT_DIR = Path(CFG["result_dir"])
RESULT_DIR.mkdir(exist_ok=True)

GENERAL_DIR = RESULT_DIR / "general"
GENERAL_DIR.mkdir(exist_ok=True)

SENTIMENT_DIR = RESULT_DIR / "sentiment"
SENTIMENT_DIR.mkdir(exist_ok=True)

_SENTIMENT_CFG = CFG.get("sentiment", {})
SENTENCE_LENGTH_LIMIT: bool = _SENTIMENT_CFG.get("sentence_length_limit", True)
LENGTH_LIMIT: int = _SENTIMENT_CFG.get("max_string_length", 500)
FILTER: bool = _SENTIMENT_CFG.get("only_sentiment_output", True)

NER_DIR = RESULT_DIR / "ner"
NER_DIR.mkdir(exist_ok=True)
NER_TAG = CFG["ner"]["ner_tag"]
NER_MINIMUM_COUNT: int = CFG["ner"]["minimum_count"]

VTT_SUMMARY_REQUIRED_COLUMNS = CFG["vtt_summuary"]["vtt_summary_req_col"]
VTT_SUMMARY_SYSTEM_INSTRUCTION_PATH = Path(CFG["vtt_summuary"]["system_instruction_path"])
VTT_SUMMARY_DIR = RESULT_DIR / "VTT_Summary"
VTT_SUMMARY_DIR.mkdir(exist_ok=True)