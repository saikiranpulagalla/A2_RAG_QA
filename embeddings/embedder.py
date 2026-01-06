"""
Embedding module for A²-RAG system.

Provides embeddings using:
1. LOCAL: HuggingFace sentence-transformers (FREE, no API calls, no quota limits) - RECOMMENDED
2. API: OpenAI or Google APIs (require API keys, have quota limits)

Supports automatic fallback between providers.
Local embeddings are fastest and most reliable for testing.
"""

from typing import Optional, Union
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL, EMBEDDING_PROVIDER
from utils import setup_logger, APIException

logger = setup_logger(__name__)

# Global embedder instance for caching
_embedder_cache: Optional[Union[OpenAIEmbeddings, GoogleGenerativeAIEmbeddings, HuggingFaceEmbeddings]] = None


def get_embedder(force_new: bool = False, use_google: bool = None, use_local: bool = True) -> Union[OpenAIEmbeddings, GoogleGenerativeAIEmbeddings, HuggingFaceEmbeddings]:
    """
    Get or create embeddings model instance.
    
    Caches embedder to avoid reinitializing multiple times.
    Falls back gracefully between providers if initialization fails.
    
    RECOMMENDED ORDER (no quota limits):
    1. LOCAL: HuggingFace sentence-transformers (FREE, no API calls)
    2. API: Google Generative AI
    3. API: OpenAI
    
    Args:
        force_new: If True, create new instance instead of using cache
        use_google: If True, use Google embeddings (API-based)
        use_local: If True, try local HuggingFace embeddings first (RECOMMENDED)
        
    Returns:
        Embeddings instance (HuggingFace, Google, or OpenAI)
        
    Raises:
        APIException: If embeddings initialization fails
        
    Example:
        >>> embedder = get_embedder()  # Uses local HuggingFace by default
        >>> vectors = embedder.embed_documents(["text1", "text2"])
    """
    global _embedder_cache
    
    if not force_new and _embedder_cache is not None:
        logger.info("Using cached embedder instance")
        return _embedder_cache
    
    # Try local HuggingFace embeddings first (NO QUOTA LIMITS!)
    if use_local:
        try:
            logger.info("Initializing LOCAL HuggingFace embeddings (all-MiniLM-L6-v2)")
            logger.info("This model is completely FREE - no API keys or quota limits needed!")
            embedder = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"}  # Use CPU (no GPU needed)
            )
            _embedder_cache = embedder
            logger.info("[OK] HuggingFace embeddings initialized successfully (no API calls needed)")
            return embedder
        except Exception as e:
            logger.warning(f"Local HuggingFace embeddings failed: {str(e)}, trying API providers...")
            # Fall through to API providers
    
    # Determine which API provider to use (Google disabled, use OpenAI only)
    # use_google parameter is deprecated; we now use OpenAI exclusively
    use_google_embeddings = False  # Always use OpenAI, Google removed from config
    
    # Try primary API provider: OpenAI
    try:
        logger.info(f"Initializing OpenAI embeddings with model: text-embedding-3-small")
        embedder = OpenAIEmbeddings(model="text-embedding-3-small")
        _embedder_cache = embedder
        logger.info("OpenAI embeddings initialized successfully")
        return embedder
    except Exception as e:
        logger.warning(f"OpenAI embeddings failed: {str(e)}, trying Google fallback...")
    
    # Try fallback API provider
    try:
        if use_google_embeddings:
            logger.info("Falling back to OpenAI embeddings...")
            embedder = OpenAIEmbeddings(model="text-embedding-3-small")
        else:
            logger.info("Falling back to Google embeddings...")
            embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        _embedder_cache = embedder
        logger.info("Fallback API embeddings initialized successfully")
        return embedder
    except Exception as fallback_error:
        logger.error(f"All embedding providers failed: {str(fallback_error)}")
        raise APIException(f"Embeddings initialization failed for all providers: {str(fallback_error)}") from fallback_error


def clear_embedder_cache():
    """
    Clear cached embedder instance.
    
    Useful for testing or switching to a new embedder.
    """
    global _embedder_cache
    _embedder_cache = None
    logger.info("Embedder cache cleared")
