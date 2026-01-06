"""
Utility module for A²-RAG system.

Provides common functionality including:
- Type definitions for Document handling
- Prompt templates
- Logging utilities
- Document processing helpers
"""

import logging
import os
import json
from datetime import datetime
from typing import Union, List
from dataclasses import dataclass
import sys


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logger(name: str, verbose: bool = False) -> logging.Logger:
    """
    Configure logger for the system.
    
    Args:
        name: Logger name (typically __name__)
        verbose: If True, log at INFO level; else WARNING level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    level = logging.INFO if verbose else logging.WARNING
    logger.setLevel(level)
    
    # Only add handler if it doesn't exist (prevent duplicate logs)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(name)s] %(levelname)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def structured_log(stage: str, info: dict, logger_name: str = None):
    """
    Emit a structured (dict-based) log message suitable for UI ingestion.

    Args:
        stage: Short stage name (Decision, Retrieval, Chunking, Generation)
        info: Dictionary with arbitrary metadata for the stage
        logger_name: Optional logger name to include
    """
    payload = {
        "stage": stage,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "logger": logger_name or "structured",
        "payload": info
    }
    # Print as JSON on stdout so Streamlit or other collectors can consume it
    try:
        print(json.dumps(payload, ensure_ascii=False))
    except Exception:
        # Fallback to simple string log
        print(f"{stage}: {info}")


# Suppress TensorFlow verbose logs (INFO/WARNING) globally to keep terminal clean
try:
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import logging as _logging
    _logging.getLogger('tensorflow').setLevel(_logging.ERROR)
except Exception:
    pass


# ============================================================================
# TYPE DEFINITIONS & DATA STRUCTURES
# ============================================================================

@dataclass
class Document:
    """
    Unified document representation.
    
    Attributes:
        content: The text content of the document
        metadata: Dictionary with source, id, or other attributes (optional)
    """
    content: str
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def page_content(self) -> str:
        """Compatibility property for LangChain Document interface."""
        return self.content


@dataclass
class RetrievalResult:
    """
    Result from retrieval operations.
    
    Attributes:
        documents: List of retrieved Document objects
        scores: Relevance scores for each document (optional)
        retrieval_stage: Either "parent" or "child" (for A²-RAG)
        num_queries: Number of queries made to vector store
    """
    documents: List[Document]
    scores: List[float] = None
    retrieval_stage: str = "single"  # "parent", "child", or "single"
    num_queries: int = 1


@dataclass
class AgentDecision:
    """
    Result from agentic decision module.
    
    Attributes:
        needs_retrieval: Boolean decision
        confidence: Confidence score [0, 1]
        reasoning: Explanation of the decision
        source: Source of decision ("llm" or "heuristic")
    """
    needs_retrieval: bool
    confidence: float
    reasoning: str = ""
    source: str = "llm"


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

RETRIEVAL_DECISION_PROMPT = """You are an intelligent assistant deciding whether external knowledge retrieval is necessary.

SKIP RETRIEVAL for:
- Common knowledge (capitals, famous people, basic facts)
- Simple definitions or explanations
- General reasoning questions
- Mathematical/logical problems
- Questions answerable from general knowledge

PERFORM RETRIEVAL for:
- Specific facts, dates, statistics
- Technical/specialized information
- Recent events or current information
- Specific people or organizations details
- Multi-hop reasoning requiring context
- Numerical comparisons or rankings

Question: {query}

Respond with exactly:
DECISION: [YES/NO]
CONFIDENCE: [0-100]
REASONING: [Brief explanation]"""

QA_PROMPT_TEMPLATE = """Answer the question using ONLY the provided context.
If the context doesn't contain relevant information, say "I cannot answer based on the provided context."

Context:
{context}

Question: {query}

Answer:"""

NAIVE_QA_PROMPT_TEMPLATE = """Answer the following question using your knowledge.

Question: {query}

Answer:"""


# ============================================================================
# DOCUMENT PROCESSING UTILITIES
# ============================================================================

def normalize_document(doc: Union[str, 'LangChainDocument', Document]) -> Document:
    """
    Convert various document formats to unified Document type.
    
    Args:
        doc: Document in string, LangChain Document, or Document format
        
    Returns:
        Normalized Document object
    """
    if isinstance(doc, Document):
        return doc
    elif isinstance(doc, str):
        return Document(content=doc)
    elif hasattr(doc, 'page_content'):  # LangChain Document
        metadata = getattr(doc, 'metadata', {})
        return Document(content=doc.page_content, metadata=metadata)
    else:
        raise TypeError(f"Unsupported document type: {type(doc)}")


def documents_to_texts(docs: List[Union[str, Document]]) -> List[str]:
    """
    Convert list of documents to list of text content.
    
    Args:
        docs: List of documents (mixed types)
        
    Returns:
        List of text strings
    """
    texts = []
    for doc in docs:
        if isinstance(doc, str):
            texts.append(doc)
        elif hasattr(doc, 'page_content'):  # LangChain Document
            texts.append(doc.page_content)
        elif isinstance(doc, Document):
            texts.append(doc.content)
        else:
            texts.append(str(doc))
    return texts


def join_documents(docs: List[Union[str, Document]], separator: str = "\n\n") -> str:
    """
    Join multiple documents into a single context string.
    
    Args:
        docs: List of documents
        separator: String to join documents with
        
    Returns:
        Concatenated context
    """
    texts = documents_to_texts(docs)
    return separator.join(texts)


# ============================================================================
# ERROR HANDLING & VALIDATION
# ============================================================================

class RAGException(Exception):
    """Base exception for RAG system errors."""
    pass


class RetrievalException(RAGException):
    """Raised when retrieval fails."""
    pass


class APIException(RAGException):
    """Raised when external API calls fail."""
    pass


def handle_empty_retrieval(docs: List[Document], fallback_msg: str = None) -> str:
    """
    Handle case where retrieval returns no documents.
    
    Args:
        docs: Retrieved documents (may be empty)
        fallback_msg: Optional fallback message
        
    Returns:
        Context string or fallback message
    """
    if not docs or len(docs) == 0:
        return fallback_msg or "[No relevant context found]"
    return join_documents(docs)
