"""
Central configuration for A2-RAG.

The defaults are chosen to make the project runnable on a laptop without paid
embedding calls: local HuggingFace embeddings are preferred, while LLMs use
OpenRouter/Gemini/OpenAI only when the relevant keys are present.
"""

# ---------------------------------------------------------------------------
# Model providers
# ---------------------------------------------------------------------------
EMBEDDING_PROVIDER = "local"  # "local", "openai", or "google"
ALLOW_EMBEDDING_PROVIDER_FALLBACK = False
EMBEDDING_CACHE_MAX_VECTORS = 5000
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_EMBEDDING_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
GOOGLE_EMBEDDING_MODEL = "models/embedding-001"
# Backward-compatible name used by older modules.
EMBEDDING_MODEL = OPENAI_EMBEDDING_MODEL

USE_OPENROUTER = True
OPENROUTER_MODEL = "openai/gpt-4o-mini"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

GEMINI_MODEL = "gemini-1.5-flash"
OPENAI_MODEL = "gpt-4o-mini"
DECISION_LLM_MODEL = OPENROUTER_MODEL
# Backward-compatible configuration name. Pipelines pass ``None`` so the
# provider factory selects a provider-compatible generation model.
LLM_MODEL = None
FALLBACK_LLM_MODEL = OPENAI_MODEL
USE_FALLBACK_MODEL = True
MAX_TOKENS = 256
LLM_TIMEOUT_SECONDS = 30
LLM_CIRCUIT_BREAKER_FAILURES = 3
LLM_CIRCUIT_BREAKER_RESET_SECONDS = 60

# Optional OpenRouter ranking metadata. Leave empty for local/evaluation use.
OPENROUTER_SITE_URL = ""
OPENROUTER_APP_TITLE = "A2-RAG-QA"

# ---------------------------------------------------------------------------
# Retry behavior
# ---------------------------------------------------------------------------
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 3
QUOTA_WAIT_SECONDS = 30

# ---------------------------------------------------------------------------
# Retrieval and chunking
# ---------------------------------------------------------------------------
NUM_DOCS = 300
TOP_K = 4
PARENT_K = 4
CHILD_K = 4
CHUNK_SIZE = 512
CHUNK_OVERLAP = 80
# Parent documents are bounded before embedding so transformer truncation does
# not make the tail of long corpus records invisible to parent retrieval.
PARENT_CHUNK_SIZE = 1000
PARENT_CHUNK_OVERLAP = 120
MAX_PARENT_INDEX_CACHE = 4
MAX_CONTEXT_CHARS = 4500
ENABLE_CONTEXT_COMPRESSION = True
MAX_CONTEXT_SENTENCES = 10

# Lexical reranking adds a cheap BM25-like safety net over dense retrieval.
ENABLE_LEXICAL_RERANK = True
LEXICAL_RERANK_WEIGHT = 0.35
MIN_RETRIEVAL_RELEVANCE = 0.02

# Hybrid retrieval and diversity control. Sparse retrieval improves exact-name,
# acronym, number, and rare-term recall; diversity reduces duplicated context.
ENABLE_SPARSE_RETRIEVAL = True
ENABLE_MMR_DIVERSITY = True
MMR_DIVERSITY_LAMBDA = 0.75
MAX_QUERY_CHARS = 1000
ENABLE_CORRECTIVE_RETRIEVAL = True
CORRECTIVE_RETRY_MULTIPLIER = 2
MAX_CORRECTIVE_VARIANTS = 6

# Set A2_RAG_OFFLINE=1 to use a deterministic extractive local fallback when
# no hosted LLM key is available. This is useful for demos/tests, not benchmark claims.
ALLOW_OFFLINE_EXTRACTIVE_LLM = True

# ---------------------------------------------------------------------------
# Agentic decision thresholds
# ---------------------------------------------------------------------------
RETRIEVAL_DECISION_CONFIDENCE_THRESHOLD = 0.55
HEURISTIC_DECISION_LOW = 0.35
HEURISTIC_DECISION_HIGH = 0.70

FORCE_RETRIEVAL_KEYWORDS = [
    "latest", "current", "today", "yesterday", "tomorrow", "recent",
    "when", "where", "who", "which", "what", "how many", "how much",
    "date", "year", "born", "died", "capital", "country", "located",
    "author", "wrote", "published", "invented", "discovered", "caused",
    "causes", "symptom", "treatment", "medicine", "virus", "disease",
    "according to", "source", "document", "paper", "report", "dataset",
]

SKIP_RETRIEVAL_KEYWORDS = [
    "opinion", "what do you think", "do you think", "brainstorm",
    "rewrite", "summarize this", "translate", "grammar", "tone",
]

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_NUM_EXAMPLES = 50

# ---------------------------------------------------------------------------
# System flags
# ---------------------------------------------------------------------------
VERBOSE = False
