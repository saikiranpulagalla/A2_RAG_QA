"""
Parent-Child Retrieval module for A²-RAG system.

Implements hierarchical retrieval with explicit late chunking:

Stage 1 (Parent Retrieval):
  - Build index from full parent documents
  - Retrieve top-K parents by similarity

Stage 2 (Late Chunking):
  - Split ONLY retrieved parents into child chunks
  - Do NOT chunk the entire corpus upfront

Stage 3 (Child Retrieval):
  - Build index from child chunks of retrieved parents
  - Retrieve top-K child chunks

This design ensures efficient retrieval while maintaining context.
"""

from typing import List, Union, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vectorstore.chroma_store import build_chroma, similarity_search_safe_chroma
from config import CHUNK_SIZE, CHUNK_OVERLAP, PARENT_K, CHILD_K
from utils import (
    setup_logger,
    Document,
    RetrievalResult,
    normalize_document,
    documents_to_texts,
    RAGException
)

logger = setup_logger(__name__)

# Cache parent store to avoid rebuilding on every query (speeds eval 2-3x)
_parent_store = None


def _chunk_documents(
    docs: Union[List[str], List[Document]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """
    Split documents into child chunks using RecursiveCharacterTextSplitter.
    
    This is explicitly called AFTER parent retrieval (late chunking), not upfront.
    
    Args:
        docs: List of documents to split
        chunk_size: Characters per chunk
        chunk_overlap: Character overlap between chunks
        
    Returns:
        List of text chunks
    """
    try:
        texts = documents_to_texts(docs)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]  # Try to preserve semantics
        )
        
        all_chunks = []
        for text in texts:
            chunks = splitter.split_text(text)
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} child chunks from {len(texts)} documents")
        return all_chunks
        
    except Exception as e:
        logger.error(f"Chunking failed: {str(e)}")
        raise RAGException(f"Document chunking failed: {str(e)}") from e


def retrieve_parents(
    query: str,
    parent_docs: Union[List[str], List[Document]],
    k: int = PARENT_K
) -> RetrievalResult:
    """
    Stage 1: Retrieve parent documents using similarity search.
    
    Builds a FAISS index from parent documents and retrieves top-K
    most similar parents for the given query.
    
    Args:
        query: Query text
        parent_docs: Full parent documents (entire sections/documents)
        k: Number of parent documents to retrieve
        
    Returns:
        RetrievalResult with retrieved parent documents
        
    Raises:
        RAGException: If retrieval fails
    """
    try:
        logger.info(f"[RETRIEVAL] Stage 1: Retrieving {k} parent documents")

        global _parent_store
        if _parent_store is None:
            logger.info("[RETRIEVAL] Building parent store (cached for session)")
            _parent_store = build_chroma(parent_docs, collection_name="parent_store")
        parent_vs = _parent_store

        # Retrieve top-K parents
        retrieved_parents = similarity_search_safe_chroma(parent_vs, query, k=k, fallback_empty=False)

        logger.info(f"[RETRIEVAL] Retrieved {len(retrieved_parents)} parent documents")
        
        return RetrievalResult(
            documents=retrieved_parents,
            retrieval_stage="parent",
            num_queries=1
        )
    except Exception as e:
        logger.error(f"Parent retrieval failed: {str(e)}")
        raise


def parent_child_retrieve(
    query: str,
    parent_docs: Union[List[str], List[Document]],
    parent_k: int = PARENT_K,
    child_k: int = CHILD_K,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> RetrievalResult:
    """
    Two-stage hierarchical retrieval with explicit late chunking.
    
    Process:
    1. Retrieve parent documents (full context)
    2. Chunk ONLY the retrieved parents (late chunking)
    3. Retrieve child chunks from parents
    
    Args:
        query: Query text
        parent_docs: Parent documents to retrieve from
        parent_k: Number of parents to retrieve in stage 1
        child_k: Number of child chunks to retrieve in stage 3
        chunk_size: Characters per child chunk
        chunk_overlap: Overlap between child chunks
        
    Returns:
        RetrievalResult with final child chunks
        
    Example:
        >>> parent_docs = [doc1, doc2, doc3, ...]
        >>> result = parent_child_retrieve(query, parent_docs)
        >>> print(f"Retrieved {len(result.documents)} child chunks")
    """
    try:
        # Stage 1: Retrieve parents
        logger.info("=== PARENT-CHILD RETRIEVAL START ===")
        parent_result = retrieve_parents(query, parent_docs, k=parent_k)
        
        if not parent_result.documents:
            logger.warning("No parent documents retrieved; cannot proceed to child retrieval")
            return RetrievalResult(
                documents=[],
                retrieval_stage="child",
                num_queries=2
            )
        
        # Stage 2: Late Chunking - Split ONLY retrieved parents
        logger.info(f"Stage 2: Chunking {len(parent_result.documents)} retrieved parents")
        child_chunks = _chunk_documents(
            parent_result.documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        if not child_chunks:
            logger.warning("No child chunks created from parents")
            return RetrievalResult(
                documents=[],
                retrieval_stage="child",
                num_queries=2
            )
        
        # Stage 3: Retrieve child chunks
        logger.info(f"Stage 3: Retrieving {child_k} child chunks from {len(child_chunks)} chunks")
        child_vs = build_chroma(child_chunks, collection_name="child_store")
        retrieved_children = similarity_search_safe_chroma(child_vs, query, k=child_k, fallback_empty=False)
        
        logger.info(f"=== RETRIEVAL COMPLETE: {len(retrieved_children)} child chunks ===")
        
        return RetrievalResult(
            documents=retrieved_children,
            retrieval_stage="child",
            num_queries=2  # Two similarity searches: parent + child
        )
        
    except Exception as e:
        logger.error(f"Parent-child retrieval failed: {str(e)}")
        raise RAGException(f"Two-stage retrieval failed: {str(e)}") from e

