"""
Vector store module for A²-RAG system.

Manages FAISS-based dense retrieval. Handles both string texts and Document objects,
with proper error handling for edge cases (empty corpus, retrieval failures).
"""

from typing import List, Union, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangChainDocument
from embeddings.embedder import get_embedder
from utils import (
    setup_logger, 
    Document, 
    RetrievalException, 
    normalize_document, 
    documents_to_texts
)

logger = setup_logger(__name__)


def build_faiss(
    docs: Union[List[str], List[Document], List[LangChainDocument]],
    index_name: str = "vector_store"
) -> FAISS:
    """
    Build a FAISS vector store from documents.
    
    Handles multiple document formats (strings, Document objects, LangChain Documents).
    Properly embeds and indexes documents for similarity search.
    
    Args:
        docs: List of documents to index (strings or Document objects)
        index_name: Name/identifier for the vector store
        
    Returns:
        FAISS vector store ready for similarity_search()
        
    Raises:
        RetrievalException: If documents are empty or embedding fails
        
    Example:
        >>> docs = ["Document 1 text", "Document 2 text"]
        >>> vs = build_faiss(docs)
        >>> results = vs.similarity_search("query", k=5)
    """
    if not docs or len(docs) == 0:
        raise RetrievalException("Cannot build FAISS index from empty document list")
    
    try:
        logger.info(f"Building FAISS index '{index_name}' with {len(docs)} documents")
        
        # Convert all documents to text strings for embedding
        texts = documents_to_texts(docs)
        
        # Get embedder and create FAISS index
        embedder = get_embedder()
        vector_store = FAISS.from_texts(texts, embedder)
        
        logger.info(f"Successfully created FAISS index with {len(texts)} documents")
        return vector_store
        
    except RetrievalException:
        raise  # Re-raise our own exceptions
    except Exception as e:
        logger.error(f"Failed to build FAISS index: {str(e)}")
        raise RetrievalException(f"FAISS indexing failed: {str(e)}") from e


def similarity_search_safe(
    vector_store: FAISS,
    query: str,
    k: int = 5,
    fallback_empty: bool = True
) -> List[Union[Document, LangChainDocument]]:
    """
    Perform similarity search with error handling.
    
    Args:
        vector_store: FAISS vector store to search
        query: Query text
        k: Number of results to retrieve
        fallback_empty: If True, return empty list on error; else raise exception
        
    Returns:
        List of retrieved documents
        
    Raises:
        RetrievalException: If search fails and fallback_empty=False
        
    Example:
        >>> results = similarity_search_safe(vs, "query text", k=5)
    """
    try:
        logger.info(f"Searching for: '{query}' (k={k})")
        results = vector_store.similarity_search(query, k=k)
        logger.info(f"Retrieved {len(results)} documents")
        return results
    except Exception as e:
        logger.warning(f"Similarity search failed: {str(e)}")
        if fallback_empty:
            logger.info("Returning empty result due to fallback setting")
            return []
        else:
            raise RetrievalException(f"Similarity search failed: {str(e)}") from e
