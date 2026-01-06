"""
Baseline RAG Pipeline for A²-RAG project.

Simple, single-stage retrieval-augmented generation system.
Serves as the comparison baseline for A²-RAG improvements.

Key Characteristics:
- ALWAYS retrieves (no agentic decision)
- EARLY chunking (chunks entire corpus upfront)
- Single-stage dense retrieval
- No parent-child hierarchies

Design:
This baseline demonstrates the standard RAG approach,
highlighting the benefits of A²-RAG's innovations.
"""

from typing import List, Union
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from vectorstore.chroma_store import build_chroma, similarity_search_safe_chroma
from utils import (
    setup_logger,
    Document,
    RAGException,
    documents_to_texts,
    handle_empty_retrieval,
    QA_PROMPT_TEMPLATE
)
from config import LLM_MODEL, FALLBACK_LLM_MODEL, USE_FALLBACK_MODEL, TOP_K, USE_OPENROUTER, OPENROUTER_MODEL, OPENROUTER_API_BASE, MAX_TOKENS

logger = setup_logger(__name__)


class BaselineRAG:
    """
    Baseline RAG system: Always retrieve, single-stage retrieval.
    
    Workflow:
    1. Index all documents upfront (early chunking)
    2. For each query, retrieve top-K documents by similarity
    3. Augment query with retrieved context
    4. Generate answer using LLM
    
    Attributes:
        vector_store: ChromaDB collection of all documents
        llm: Language model for QA
    """
    
    def __init__(
        self,
        documents: Union[List[str], List[Document]],
        model: str = LLM_MODEL,
        k: int = TOP_K
    ):
        """
        Initialize Baseline RAG with documents.
        
        Args:
            documents: List of documents to index (strings or Document objects)
            model: LLM model to use for answer generation
            k: Number of documents to retrieve per query
            
        Raises:
            RAGException: If document indexing fails
        """
        self.k = k
        import os
        # Primary LLM: prefer OpenRouter (if configured), then Gemini, then OpenAI fallback
        self.llm = None
        self.fallback_llm = None

        openrouter_key = os.getenv('OPENROUTER_API_KEY')
        openrouter_base = OPENROUTER_API_BASE or os.getenv('OPENROUTER_API_BASE')
        # If user provided an OpenRouter key but not a base, assume the common OpenRouter API base.
        if USE_OPENROUTER and openrouter_key and not openrouter_base:
            default_openrouter_base = "https://api.openrouter.ai/v1"
            openrouter_base = default_openrouter_base
            logger.info("OPENROUTER_API_KEY present; using default OPENROUTER_API_BASE=https://api.openrouter.ai/v1")

        try_openrouter = USE_OPENROUTER and (openrouter_key is not None) and (openrouter_base)
        if try_openrouter:
            prev_base = os.environ.get('OPENAI_API_BASE')
            prev_key = os.environ.get('OPENAI_API_KEY')
            os.environ['OPENAI_API_BASE'] = openrouter_base
            os.environ['OPENAI_API_KEY'] = openrouter_key
            try:
                self.llm = ChatOpenAI(model=OPENROUTER_MODEL, temperature=0)
                logger.info(f"Primary LLM (OpenRouter) initialized: {OPENROUTER_MODEL}")
                # Initialize OpenAI fallback if configured so we can switch when OpenRouter fails
                if USE_FALLBACK_MODEL and FALLBACK_LLM_MODEL:
                    try:
                        self.fallback_llm = ChatOpenAI(model=FALLBACK_LLM_MODEL, temperature=0)
                        logger.info(f"Initialized fallback OpenAI model: {FALLBACK_LLM_MODEL}")
                    except Exception as e:
                        logger.warning(f"Failed to initialize fallback OpenAI model: {e}")
            except Exception as e:
                logger.warning(f"OpenRouter init failed: {e}")
                self.llm = None
            finally:
                if prev_base is not None:
                    os.environ['OPENAI_API_BASE'] = prev_base
                else:
                    os.environ.pop('OPENAI_API_BASE', None)
                if prev_key is not None:
                    os.environ['OPENAI_API_KEY'] = prev_key
                else:
                    os.environ.pop('OPENAI_API_KEY', None)

        if self.llm is None:
            gemini_candidates = [model, "gemini-2.5-flash", "gemini-1.5-flash"]
            for gm in gemini_candidates:
                if not gm:
                    continue
                try:
                    self.llm = ChatGoogleGenerativeAI(model=gm, temperature=0)
                    logger.info(f"Primary LLM (Gemini) initialized: {gm}")
                    # Initialize OpenAI fallback if configured so we can switch when Gemini is rate-limited
                    if USE_FALLBACK_MODEL and FALLBACK_LLM_MODEL:
                        try:
                            self.fallback_llm = ChatOpenAI(model=FALLBACK_LLM_MODEL, temperature=0)
                            logger.info(f"Initialized fallback OpenAI model: {FALLBACK_LLM_MODEL}")
                        except Exception as e:
                            logger.warning(f"Failed to initialize fallback OpenAI model: {e}")
                    break
                except Exception as e:
                    logger.warning(f"Primary Gemini init failed for {gm}: {e}")

        if self.llm is None and USE_FALLBACK_MODEL and FALLBACK_LLM_MODEL:
            try:
                self.llm = ChatOpenAI(model=FALLBACK_LLM_MODEL, temperature=0)
                logger.info(f"Using OpenAI fallback model: {FALLBACK_LLM_MODEL}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI fallback model: {e}")
                raise

        if self.llm is None:
            raise RAGException("No LLM available: OpenRouter/Gemini candidates failed and no fallback available")
        
        try:
            logger.info(f"Initializing BaselineRAG with {len(documents)} documents")
            self.vector_store = build_chroma(documents, collection_name="baseline_store")
            logger.info(f"BaselineRAG initialized successfully with {self.vector_store.count()} documents indexed")
        except Exception as e:
            logger.error(f"Failed to initialize BaselineRAG: {str(e)}")
            raise RAGException(f"BaselineRAG initialization failed: {str(e)}") from e
    
    def retrieve(self, query: str) -> List[Union[str, Document]]:
        """
        Retrieve top-K documents for the query.
        
        Args:
            query: Input question
            
        Returns:
            List of retrieved documents
        """
        try:
            logger.info(f"[RETRIEVAL] Retrieving {self.k} documents for: '{query}'")
            docs = similarity_search_safe_chroma(
                self.vector_store,
                query,
                k=self.k,
                fallback_empty=True
            )
            logger.info(f"[RETRIEVAL] Retrieved {len(docs)} documents")
            return docs
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            return []
    
    def answer(self, query: str, return_context: bool = False) -> Union[str, dict]:
        """
        Generate answer for the query using retrieved context.
        
        Process:
        1. Retrieve relevant documents
        2. Construct context from retrieval
        3. Generate answer using LLM
        
        Args:
            query: Input question
            return_context: If True, return dict with answer + context
            
        Returns:
            Generated answer string, or dict with answer + metadata
            
        Example:
            >>> rag = BaselineRAG(documents)
            >>> answer = rag.answer("What is X?")
            >>> print(answer)
        """
        try:
            # Stage 1: Retrieve
            docs = self.retrieve(query)
            
            # Stage 2: Build context
            if not docs:
                context = "[No relevant documents found]"
                logger.warning("No documents retrieved for query")
            else:
                context = handle_empty_retrieval(docs)
            
            # Stage 3: Generate answer
            prompt = QA_PROMPT_TEMPLATE.format(
                context=context,
                query=query
            )
            
            try:
                logger.info("[GENERATION] Generating answer with LLM")
                response = self.llm.invoke(prompt, max_tokens=MAX_TOKENS)
                answer = response.content.strip()
            except Exception as e:
                # Fallback to OpenAI if Gemini fails
                if self.fallback_llm:
                    logger.warning(f"Primary LLM failed ({str(e)}), using fallback OpenAI")
                    try:
                        response = self.fallback_llm.invoke(prompt, max_tokens=MAX_TOKENS)
                        answer = response.content.strip()
                    except Exception as fallback_e:
                        raise RAGException(f"Both primary and fallback LLM failed: {str(fallback_e)}")
                else:
                    raise RAGException(f"LLM failed and no fallback available: {str(e)}")
            
            logger.info(f"[GENERATION] Generated answer ({len(answer)} chars)")
            
            if return_context:
                return {
                    "answer": answer,
                    "context": context,
                    "num_docs_retrieved": len(docs),
                    "model": LLM_MODEL
                }
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            fallback = f"Error generating answer: {str(e)}"
            if return_context:
                return {"answer": fallback, "context": "", "num_docs_retrieved": 0, "model": LLM_MODEL}
            return fallback

