"""
Late Chunking utilities for A²-RAG system.

Provides helper functions to process retrieved documents
(which are already chunks from parent-child retrieval).

Key Concept:
- Chunking happens AFTER parent retrieval (late chunking)
- This module handles the final context assembly
- Not the primary chunking mechanism (see parent_child_retrieval.py)
"""

from typing import List, Union
from utils import setup_logger, Document, handle_empty_retrieval, documents_to_texts

logger = setup_logger(__name__)


def extract_context(
    docs: Union[List[str], List[Document]],
    separator: str = "\n\n",
    max_length: int = None
) -> str:
    """
    Extract and join context from retrieved documents.
    
    Converts list of documents/chunks into a single context string
    suitable for the LLM's prompt.
    
    Args:
        docs: Retrieved documents or chunks (strings or Document objects)
        separator: String to join documents with
        max_length: Optional max length (truncate if exceeded)
        
    Returns:
        Joined context string, or empty/fallback if no docs
        
    Example:
        >>> docs = [doc1, doc2, doc3]
        >>> context = extract_context(docs)
        >>> print(len(context))
    """
    try:
        if not docs or len(docs) == 0:
            logger.warning("No documents provided for context extraction")
            return handle_empty_retrieval([])
        
        # Convert documents to texts
        texts = documents_to_texts(docs)
        context = separator.join(texts)
        
        # Optionally truncate
        if max_length and len(context) > max_length:
            logger.info(f"Truncating context from {len(context)} to {max_length} chars")
            context = context[:max_length] + "...[truncated]"
        
        logger.info(f"Extracted context: {len(context)} characters from {len(docs)} documents")
        return context
        
    except Exception as e:
        logger.error(f"Context extraction failed: {str(e)}")
        return handle_empty_retrieval([], fallback_msg=f"[Error extracting context: {str(e)}]")


def prepare_context_for_qa(
    docs: Union[List[str], List[Document]],
    include_source_info: bool = False
) -> str:
    """
    Prepare context specifically formatted for QA tasks.
    
    Can optionally include document source/metadata information.
    
    Args:
        docs: Retrieved documents
        include_source_info: If True, add source metadata
        
    Returns:
        Formatted context string
    """
    try:
        if not docs:
            return "[No context available]"
        
        context_parts = []
        for i, doc in enumerate(docs, 1):
            if hasattr(doc, 'content'):  # Document object
                text = doc.content
                metadata = getattr(doc, 'metadata', {})
            elif hasattr(doc, 'page_content'):  # LangChain Document
                text = doc.page_content
                metadata = getattr(doc, 'metadata', {})
            else:  # String
                text = str(doc)
                metadata = {}
            
            if include_source_info and metadata:
                source = metadata.get('source', 'unknown')
                context_parts.append(f"[Source {i}: {source}]\n{text}")
            else:
                context_parts.append(text)
        
        return "\n\n".join(context_parts)
        
    except Exception as e:
        logger.error(f"Context preparation failed: {str(e)}")
        return f"[Error preparing context: {str(e)}]"

