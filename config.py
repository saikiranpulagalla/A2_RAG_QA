"""
Configuration module for A²-RAG (Adaptive & Agentic Retrieval-Augmented Generation).

Centralizes all hyperparameters and model configurations for both Baseline RAG and A²-RAG systems.
Ensures reproducibility and easy tuning of key parameters.

Key Sections:
- LLM & Embedding Models
- Chunking Parameters
- Retrieval Parameters
- Evaluation Metrics
- System Behavior Flags
"""

# ============================================================================
# LLM & EMBEDDING MODEL CONFIGURATION
# ============================================================================
# Models used for both systems. Update these to switch providers/versions.
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding model
EMBEDDING_PROVIDER = "openai"  # "openai" only
USE_OPENAI_EMBEDDINGS = True  # Use OpenAI embeddings

# LLM Strategy: Optimize cost + quality using OpenRouter
# Primary: OpenRouter (openai/gpt-3.5-turbo) - cost-effective, multi-provider routing
# Fallback: Direct OpenAI (gpt-3.5-turbo) - backup if OpenRouter unavailable
DECISION_LLM_MODEL = "openai/gpt-3.5-turbo"  # Decision module via OpenRouter
LLM_MODEL = "openai/gpt-3.5-turbo"  # Generation model via OpenRouter
FALLBACK_LLM_MODEL = "gpt-3.5-turbo"  # Fallback: Direct OpenAI API
USE_FALLBACK_MODEL = False  # Disable fallback (OpenRouter is primary)
MAX_TOKENS = 512  # Max tokens for LLM response (limited to prevent quota issues)

# ============================================================================
# RETRY & QUOTA RESILIENCE
# ============================================================================
MAX_RETRIES = 3  # Max retry attempts for API calls
RETRY_BACKOFF_SECONDS = 5  # Initial backoff time before retry (increases exponentially)
QUOTA_WAIT_SECONDS = 60  # Wait time when quota exceeded error detected

# ============================================================================
# CHUNKING PARAMETERS (Late Chunking in A²-RAG)
# ============================================================================
# Applied AFTER parent retrieval in A²-RAG to split retrieved documents into child chunks.
# Not used in Baseline RAG (which chunks upfront - early chunking).
CHUNK_SIZE = 512  # Characters per chunk. Smaller = more granular, larger = more context
CHUNK_OVERLAP = 50  # Characters. Prevents loss of info at chunk boundaries

# ============================================================================
# DOCUMENT CORPUS PARAMETERS
# ============================================================================
# Number of documents to load from Wikipedia for indexing.
# Larger corpus improves hit rate and retrieval quality, but increases indexing time.
# Available: 1000 documents in wiki_docs.json
# Recommended range: 100-500 documents for good quality/speed balance
NUM_DOCS = 300  # Number of documents to load (increased from 10 for better retrieval quality)

# ============================================================================
# RETRIEVAL PARAMETERS
# ============================================================================
# Top-K for both parent and child retrieval in A²-RAG.
# Baseline RAG uses this for single-stage dense retrieval.
# Reduced to 3 for token budget management (prevents 4096 token errors).
TOP_K = 3  # Number of top documents/chunks to retrieve
PARENT_K = 3  # Number of parent documents to retrieve (A²-RAG stage 1)
CHILD_K = 3   # Number of child chunks from parents (A²-RAG stage 2)

# ============================================================================
# AGENTIC DECISION PARAMETERS (A²-RAG Only)
# ============================================================================
# Thresholds for deciding whether retrieval is necessary.
# Calibrated to maximize retrieval for factual questions while skipping opinion/reasoning-only.
# TUNED: Lowered threshold from 0.5 to 0.35 for better medium-to-high confidence triggering

RETRIEVAL_DECISION_CONFIDENCE_THRESHOLD = 0.35  # Skip retrieval only if confidence < 0.35 (medium-to-high confidence triggers retrieval)

# Keywords that FORCE retrieval (factual, entity-based, definition, historical, medical)
FORCE_RETRIEVAL_KEYWORDS = [
    "where", "when", "who", "what", "which",  # WH-questions (factual)
    "define", "definition",  # Definitions
    "causes", "caused",  # Causation (factual/medical)
    "lives in", "located in", "capital", "country",  # Entity locations
    "born", "died", "year", "date",  # Historical facts
    "disease", "symptom", "treatment", "medicine", "virus",  # Medical
    "discovered", "invented", "created",  # Historical discovery
    "author", "wrote", "published"  # Literary/historical
]

# Keywords that suggest SKIP retrieval (opinion, reasoning, conversational)
SKIP_RETRIEVAL_KEYWORDS = [
    "think", "believe", "opinion", "agree", "disagree",  # Opinion
    "should", "would", "could",  # Hypothetical
    "why do you", "what do you think",  # Conversational
    "explain", "reason"  # Pure reasoning (no facts)
]

# Heuristic decision thresholds
HEURISTIC_DECISION_LOW = 0.35   # below -> likely skip retrieval
HEURISTIC_DECISION_HIGH = 0.65  # above -> likely retrieve
# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================
# Metrics configuration for comparing Baseline vs A²-RAG
EVAL_NUM_EXAMPLES = 50  # Use 20 QA examples to reduce API quota pressure (was 50)
EVALUATION_METRICS = ["exact_match", "f1", "retrieval_hit_rate"]  # Metrics to compute
TOP_K_FOR_RETRIEVAL_HIT = 5  # For evaluating if correct document was in top-K
SAVE_EVAL_DETAILS = True  # Save per-query evaluation details to CSV
LOG_DECISION_DETAILS = True  # Log retrieval decision, confidence, and reasoning for A²-RAG

# ============================================================================
# SYSTEM BEHAVIOR FLAGS
# ============================================================================
VERBOSE = True  # Print intermediate steps for debugging
DETERMINISTIC = True  # Set random seeds for reproducibility
DEBUG_MODE = False  # Log detailed info on agent decisions and retrieval

# ============================================================================
# OPENROUTER CONFIGURATION (Primary LLM Provider)
# ============================================================================
# OpenRouter provides access to multiple LLM providers with unified API
# Cost-effective routing and fallback support across providers
USE_OPENROUTER = True  # Enable OpenRouter as primary LLM provider
OPENROUTER_MODEL = "openai/gpt-3.5-turbo"  # Model format: "provider/model-name"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"  # OpenRouter API endpoint
OPENROUTER_API_KEY = None  # Set via environment variable OPENROUTER_API_KEY


# ============================================================================
# DATASET PARAMETERS
# ============================================================================
DATASET_SIZE = 1000  # Max questions to evaluate (for full evaluation)
EVAL_SAMPLE_SIZE = 20  # Sample size for quick validation / UI-triggered runs
