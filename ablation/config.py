from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "ablation" / "v1"
CACHE_DIR = ROOT / ".cache" / "ablation" / "v1"
RECORDS_DIR = CACHE_DIR / "records"
LLM_CACHE_DIR = CACHE_DIR / "llm_responses"
RETRIEVAL_CACHE_DIR = CACHE_DIR / "retrieval_checkpoints"
TRAINING_CACHE_DIR = CACHE_DIR / "training_checkpoints"
QUERY_EMBEDDING_CACHE = CACHE_DIR / "query_embeddings.sqlite3"
SUBSET_PATH = ROOT / "Dataset" / "Ablation_Subset" / "ablation_subset_seed42.xlsx"
RAW_CHROMA_PATH = ROOT / "chroma_db_hpo_260623"
PROJECTED_CHROMA_PATH = ROOT / "chroma_db_hpo_oar_ablation_260623"
MAIN_RUN_DIR = (
    ROOT
    / "results"
    / "rerank"
    / ".runs"
    / "gpt-5.6-sol_cd8b26bab0b5"
)

DATASET_ORDER = (
    "genereviews-10",
    "id-68",
    "gsc2017",
    "gsc2024",
    "csc",
    "fgdd_phenotype",
    "bc8_t3",
)
MAIN_DATASET_KEYS = {
    "genereviews-10": "genereviews-10",
    "id-68": "id-68",
    "gsc2017": "gsc2017",
    "gsc2024": "gsc2024",
    "csc": "csc",
    "fgdd_phenotype": "fgdd",
    "bc8_t3": "bc8_t3",
}
SUBSET_DATASET_KEYS = {
    "genereviews-10": "genereviews_10",
    "id-68": "id_68",
    "gsc2017": "gsc2017",
    "gsc2024": "gsc2024",
    "csc": "csc",
    "fgdd_phenotype": "fgdd_phenotype",
    "bc8_t3": "bc8_t3",
}

SEED = 42
SUBSET_ROWS_PER_DATASET = 300
FULL_SAMPLE_COUNT = 13_390
SUBSET_SAMPLE_COUNT = 2_100
CANDIDATE_K_VALUES = (1, 3, 5, 10, 20)
RETRIEVAL_TOP_K = 20
MAX_CONCURRENCY = 10
REQUESTS_PER_MINUTE = 200
MAX_ATTEMPTS = 3
PAUSE_ON_429_SECONDS = 60

OAR_EPOCHS = 6
OAR_CHECKPOINT_EPOCH = 5
OAR_BATCH_SIZE = 64
OAR_LEARNING_RATE = 1e-5
OAR_TEMPERATURE = 0.05
OAR_WEIGHT_DECAY = 1e-5
OAR_IDENTITY_REGULARIZATION = 1e-6
OAR_GRADIENT_CLIP = 1.0
OAR_HARD_NEGATIVE_QUERY_K = 50
OAR_NEGATIVE_CONCEPTS = 10


@dataclass(frozen=True)
class BackboneSpec:
    key: str
    model_name: str
    collection_name: str
    backend: str
    revision: str
    pooling: str
    dimension: int
    batch_size: int
    config_section: str | None = None
    train_oar: bool = True


BACKBONES = (
    BackboneSpec(
        "text_embedding_3_large",
        "text-embedding-3-large",
        "hpo_text_embedding_3_large_20260623",
        "openai",
        "provider-managed model ID (immutable revision unavailable)",
        "provider embedding endpoint",
        3072,
        512,
        "embedding",
        True,
    ),
    BackboneSpec(
        "text_embedding_3_small",
        "text-embedding-3-small",
        "hpo_text_embedding_3_small_20260623",
        "openai",
        "provider-managed model ID (immutable revision unavailable)",
        "provider embedding endpoint",
        1536,
        128,
        "embedding_small",
    ),
    BackboneSpec(
        "biolord",
        "FremyCompany/BioLORD-2023",
        "hpo_biolord_20260623",
        "sentence_transformers",
        "167aab527b238a50ca65224e6319215d2ff4fc9f",
        "model-native SentenceTransformer pooling",
        768,
        128,
    ),
    BackboneSpec(
        "biobert",
        "dmis-lab/biobert-base-cased-v1.2",
        "hpo_biobert_base_cased_v1_2_mean_pooling_20260623",
        "transformers_mean_pooling",
        "67c9c25b46986521ca33df05d8540da1210b3256",
        "attention-mask mean pooling of last_hidden_state",
        768,
        128,
    ),
    BackboneSpec(
        "pubmedbert",
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "hpo_pubmedbert_abstract_fulltext_mean_pooling_20260623",
        "transformers_mean_pooling",
        "e1354b7a3a09615f6aba48dfad4b7a613eef7062",
        "attention-mask mean pooling of last_hidden_state",
        768,
        128,
    ),
    BackboneSpec(
        "clinicalbert",
        "emilyalsentzer/Bio_ClinicalBERT",
        "hpo_clinicalbert_mean_pooling_20260623",
        "transformers_mean_pooling",
        "d5892b39a4adaed74b92212a44081509db72f87b",
        "attention-mask mean pooling of last_hidden_state",
        768,
        128,
    ),
)
BACKBONE_BY_KEY = {spec.key: spec for spec in BACKBONES}


EXPERIMENT_FILES = {
    "E1": RESULTS_DIR / "E1_raw_te3l_full.xlsx",
    "E2": RESULTS_DIR / "E2_oar_only_full.xlsx",
    "E3": RESULTS_DIR / "E3_oar_lcr_full.xlsx",
    "E4": RESULTS_DIR / "E4_claude_opus_5_subset.xlsx",
    "E5": RESULTS_DIR / "E5_embedding_oar_subset.xlsx",
    "E6": RESULTS_DIR / "E6_candidate_k_subset.xlsx",
}

E4_LLM_CONFIG_KEYS = {
    "claude-opus-5": "llm2",
    "deepseek-v4-flash": "deepseek-v4-flash",
    "deepseek-v4-pro": "deepseek-v4-pro",
}
E4_LLM_FILES = {
    "claude-opus-5": EXPERIMENT_FILES["E4"],
    "deepseek-v4-flash": RESULTS_DIR / "E4_deepseek_v4_flash_subset.xlsx",
    "deepseek-v4-pro": RESULTS_DIR / "E4_deepseek_v4_pro_subset.xlsx",
}


def ensure_directories() -> None:
    for path in (
        RESULTS_DIR,
        CACHE_DIR,
        RECORDS_DIR,
        LLM_CACHE_DIR,
        RETRIEVAL_CACHE_DIR,
        TRAINING_CACHE_DIR,
        RESULTS_DIR / "oar_models",
    ):
        path.mkdir(parents=True, exist_ok=True)
