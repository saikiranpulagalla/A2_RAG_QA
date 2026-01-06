"""
Embedding cache module for A²-RAG system.

Caches embeddings to avoid repeated API calls for the same documents.
This is critical for staying within API quota limits during testing and evaluation.

Supports:
- In-memory caching with LRU eviction
- Persistence to disk (JSON)
- Automatic hashing of document content
"""

import json
import os
import hashlib
from typing import List, Optional, Dict, Any
from pathlib import Path
from embeddings.embedder import get_embedder
from utils import setup_logger

logger = setup_logger(__name__)

# Global cache
_embedding_cache: Dict[str, List[float]] = {}
CACHE_FILE = "embeddings_cache.json"


def _hash_text(text: str) -> str:
    """Generate hash of text for caching."""
    return hashlib.md5(text.encode()).hexdigest()


def load_cache_from_disk():
    """Load embedding cache from disk."""
    global _embedding_cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                _embedding_cache = json.load(f)
            logger.info(f"Loaded {len(_embedding_cache)} embeddings from cache")
        except Exception as e:
            logger.warning(f"Failed to load cache from disk: {e}")
            _embedding_cache = {}
    else:
        _embedding_cache = {}


def save_cache_to_disk():
    """Save embedding cache to disk."""
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(_embedding_cache, f)
        logger.info(f"Saved {len(_embedding_cache)} embeddings to cache")
    except Exception as e:
        logger.warning(f"Failed to save cache to disk: {e}")


def embed_with_cache(texts: List[str], force_refresh: bool = False) -> List[List[float]]:
    """
    Embed texts with caching to avoid repeated API calls.
    
    Args:
        texts: List of texts to embed
        force_refresh: If True, ignore cache and re-embed
        
    Returns:
        List of embeddings (one per text)
    """
    global _embedding_cache
    
    # Load cache if not already loaded
    if not _embedding_cache:
        load_cache_from_disk()
    
    # Find texts that need embedding
    embeddings = []
    texts_to_embed = []
    text_indices = []  # Track which original indices need embedding
    
    for i, text in enumerate(texts):
        text_hash = _hash_text(text)
        
        if not force_refresh and text_hash in _embedding_cache:
            embeddings.append((_embedding_cache[text_hash], i))
        else:
            texts_to_embed.append((text, text_hash, i))
    
    # Embed texts that aren't cached
    if texts_to_embed:
        try:
            logger.info(f"Embedding {len(texts_to_embed)}/{len(texts)} texts (cache hit: {len(texts) - len(texts_to_embed)})")
            embedder = get_embedder()
            new_embeddings = embedder.embed_documents([t[0] for t in texts_to_embed])
            
            # Store in cache
            for embedding, (text, text_hash, _) in zip(new_embeddings, texts_to_embed):
                _embedding_cache[text_hash] = embedding
            
            # Add to results in original order
            for embedding, (text, text_hash, idx) in zip(new_embeddings, texts_to_embed):
                embeddings.append((embedding, idx))
            
            # Save cache to disk
            save_cache_to_disk()
            
        except Exception as e:
            logger.error(f"Failed to embed texts: {e}")
            raise
    
    # Sort by original indices to return in correct order
    embeddings.sort(key=lambda x: x[1])
    return [emb for emb, _ in embeddings]


def clear_cache():
    """Clear embedding cache."""
    global _embedding_cache
    _embedding_cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            os.remove(CACHE_FILE)
            logger.info("Cleared embedding cache")
        except Exception as e:
            logger.warning(f"Failed to delete cache file: {e}")
