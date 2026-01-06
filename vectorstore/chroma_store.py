"""
ChromaDB vector store module for A²-RAG system.

Manages ChromaDB-based dense retrieval with embedding caching to avoid quota exhaustion.
ChromaDB is more efficient than FAISS:
- Lazy embedding (embeds on demand, not upfront)
- Automatic persistence
- Better handling of large document sets

With embedding cache:
- Reuses embeddings for same documents
- Minimizes API calls
- Dramatically reduces quota usage during testing

Handles both string texts and Document objects with proper error handling.
"""

from typing import List, Union, Optional
import chromadb
from chromadb.config import Settings
from chromadb import EmbeddingFunction, Documents, Embeddings
from langchain_core.documents import Document as LangChainDocument
from embeddings.embedding_cache import embed_with_cache
from utils import (
    setup_logger, 
    Document, 
    RetrievalException, 
    normalize_document, 
    documents_to_texts
)

logger = setup_logger(__name__)

# Global ChromaDB client for caching
_chroma_client = None


class CachedEmbeddingFunction(EmbeddingFunction):
    """Wrapper to use cached embeddings with ChromaDB."""
    
    def __call__(self, input: Documents) -> Embeddings:
        """Embed documents using cached embeddings."""
        try:
            embeddings = embed_with_cache(input)
            return embeddings
        except Exception as e:
            logger.error(f"Embedding failed: {str(e)}")
            raise


def get_chroma_client():
    """Get or create ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        # Use ephemeral (in-memory) ChromaDB for simplicity
        _chroma_client = chromadb.EphemeralClient()
    return _chroma_client


def build_chroma(
    docs: Union[List[str], List[Document], List[LangChainDocument]],
    collection_name: str = "documents",
    embedding_function=None
) -> chromadb.Collection:
    """
    Build a ChromaDB collection from documents.
    
    ChromaDB only embeds documents when they're added, not upfront.
    This makes it much more efficient with API quota limits.
    
    Handles multiple document formats (strings, Document objects, LangChain Documents).
    
    Args:
        docs: List of documents to index (strings or Document objects)
        collection_name: Name of the ChromaDB collection
        embedding_function: Custom embedding function (uses LangChain embeddings by default)
        
    Returns:
        ChromaDB collection ready for querying
        
    Raises:
        RetrievalException: If documents are empty
        
    Example:
        >>> docs = ["Document 1 text", "Document 2 text"]
        >>> collection = build_chroma(docs, collection_name="my_docs")
        >>> results = collection.query(query_texts=["query"], n_results=5)
    """
    if not docs or len(docs) == 0:
        raise RetrievalException("Cannot build ChromaDB collection from empty document list")
    
    try:
        logger.info(f"Building ChromaDB collection '{collection_name}' with {len(docs)} documents")
        
        # Convert all documents to text strings
        texts = documents_to_texts(docs)
        
        # Get ChromaDB client
        client = get_chroma_client()
        
        # Use cached embeddings by default
        if embedding_function is None:
            embedding_function = CachedEmbeddingFunction()
        
        # Create or get collection
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Clear existing documents in collection (if any)
        if collection.count() > 0:
            all_ids = collection.get()["ids"]
            if all_ids:
                collection.delete(ids=all_ids)
        
        # Add documents with IDs
        ids = [f"doc_{i}" for i in range(len(texts))]
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=[{"index": i} for i in range(len(texts))]
        )
        
        # Verify all documents were indexed
        indexed_count = collection.count()
        expected_count = len(texts)
        logger.info(f"Successfully added {len(texts)} documents to ChromaDB collection '{collection_name}'")
        if indexed_count != expected_count:
            logger.warning(f"Document count mismatch! Expected {expected_count}, got {indexed_count}")
        else:
            logger.info(f"Verification OK: All {indexed_count} documents indexed successfully")
        
        return collection
        
    except RetrievalException:
        raise  # Re-raise our own exceptions
    except Exception as e:
        logger.error(f"Failed to build ChromaDB collection: {str(e)}")
        raise RetrievalException(f"ChromaDB indexing failed: {str(e)}") from e


def similarity_search_chroma(
    collection: chromadb.Collection,
    query: str,
    k: int = 5,
    fallback_empty: bool = True
) -> List[str]:
    """
    Perform similarity search on ChromaDB collection.
    
    Args:
        collection: ChromaDB collection to search
        query: Query text
        k: Number of results to retrieve
        fallback_empty: If True, return empty list on error; else raise exception
        
    Returns:
        List of retrieved document texts
        
    Raises:
        RetrievalException: If search fails and fallback_empty=False
        
    Example:
        >>> results = similarity_search_chroma(collection, "query text", k=5)
    """
    try:
        logger.info(f"ChromaDB similarity search for: '{query}' (k={k})")
        
        # Query the collection
        results = collection.query(
            query_texts=[query],
            n_results=k
        )
        
        # Extract document texts from results
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        logger.info(f"Retrieved {len(documents)} documents from ChromaDB")
        
        # Return as list of strings (ChromaDB stores text directly)
        return documents if documents else []
        
    except Exception as e:
        if fallback_empty:
            logger.warning(f"ChromaDB search failed: {str(e)}, returning empty results")
            return []
        else:
            logger.error(f"ChromaDB search failed: {str(e)}")
            raise RetrievalException(f"ChromaDB similarity search failed: {str(e)}") from e


def similarity_search_safe_chroma(
    collection: chromadb.Collection,
    query: str,
    k: int = 5,
    fallback_empty: bool = True
) -> List[str]:
    """
    Wrapper for similarity_search_chroma with consistent error handling.
    
    Args:
        collection: ChromaDB collection to search
        query: Query text
        k: Number of results to retrieve
        fallback_empty: If True, return empty list on error; else raise exception
        
    Returns:
        List of retrieved document texts
    """
    return similarity_search_chroma(collection, query, k, fallback_empty)


def clear_chroma_cache():
    """
    Clear cached ChromaDB client.
    
    Useful for testing or resetting the vector store.
    """
    global _chroma_client
    _chroma_client = None
    logger.info("ChromaDB client cache cleared")
