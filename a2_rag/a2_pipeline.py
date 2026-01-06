"""
A²-RAG (Adaptive & Agentic Retrieval-Augmented Generation) Pipeline.

Core system implementing adaptive retrieval decisions and hierarchical
parent-child retrieval with explicit late chunking.

Key Features:
- Agentic retrieval decision (decides when to retrieve)
- Parent-child hierarchical retrieval (two-stage dense retrieval)
- Late chunking (chunk AFTER parent retrieval, not before)
- Explainability (returns decision rationale and metadata)

Design Philosophy:
This system demonstrates that NOT all queries need expensive retrieval,
and hierarchical retrieval can improve efficiency and relevance.
"""

from typing import List, Union, Optional, Dict, Any, Tuple
import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from a2_rag.agent_decision import needs_retrieval, AgentDecision
from a2_rag.parent_child_retrieval import parent_child_retrieve, RetrievalResult
from a2_rag.late_chunking import extract_context, prepare_context_for_qa
from config import LLM_MODEL, FALLBACK_LLM_MODEL, USE_FALLBACK_MODEL, USE_OPENROUTER, OPENROUTER_MODEL, OPENROUTER_API_BASE, MAX_TOKENS
from utils import (
    setup_logger,
    Document,
    RAGException,
    QA_PROMPT_TEMPLATE,
    NAIVE_QA_PROMPT_TEMPLATE,
    APIException
)
logger = setup_logger(__name__)


class A2RAG:
    """
    Adaptive & Agentic Retrieval-Augmented Generation System.
    
    Workflow:
    1. Agent Decision: Determines if retrieval is necessary
    2. If NO retrieval needed: Answer directly using LLM knowledge
    3. If retrieval needed:
       a. Parent Retrieval: Get top-K parent documents
       b. Late Chunking: Split only retrieved parents
       c. Child Retrieval: Get top-K child chunks from parents
    4. Answer Generation: Create response using context (or direct)
    
    Attributes:
        documents: Parent documents to retrieve from
        llm: Language model for QA and decision-making
    """
    
    def __init__(
        self,
        documents: Union[List[str], List[Document]],
        model: str = LLM_MODEL
    ):
        """
        Initialize A²-RAG system.
        
        Args:
            documents: Parent documents (full sections/documents)
            model: LLM model for QA (OpenAI gpt-4o-mini for generation)
            
        Raises:
            RAGException: If initialization fails
        """
        if not documents or len(documents) == 0:
            raise RAGException("A2RAG requires non-empty document list")
        
        self.documents = documents
        # Primary LLM: prefer OpenRouter (if configured), then Gemini, then OpenAI fallback
        self.llm = None
        self.fallback_llm = None

        # Try OpenRouter only if enabled and BOTH API key + API base present
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
            # Use OpenRouter API base and key from config/env
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
                # restore env to previous values if they existed
                if prev_base is not None:
                    os.environ['OPENAI_API_BASE'] = prev_base
                else:
                    os.environ.pop('OPENAI_API_BASE', None)
                if prev_key is not None:
                    os.environ['OPENAI_API_KEY'] = prev_key
                else:
                    os.environ.pop('OPENAI_API_KEY', None)

        # If OpenRouter not used or failed, try Gemini candidates
        if self.llm is None:
            gemini_candidates = [model, "gemini-2.5-flash", "gemini-1.5-flash"]
            for gm in gemini_candidates:
                if not gm:
                    continue
                try:
                    self.llm = ChatGoogleGenerativeAI(model=gm, temperature=0)
                    logger.info(f"Primary LLM (Gemini) initialized: {gm}")
                    # If configured, initialize an OpenAI fallback for generation
                    if USE_FALLBACK_MODEL and FALLBACK_LLM_MODEL:
                        try:
                            self.fallback_llm = ChatOpenAI(model=FALLBACK_LLM_MODEL, temperature=0)
                            logger.info(f"Initialized fallback OpenAI model: {FALLBACK_LLM_MODEL}")
                        except Exception as e:
                            logger.warning(f"Failed to initialize fallback OpenAI model: {e}")
                    break
                except Exception as e:
                    logger.warning(f"Primary Gemini init failed for {gm}: {e}")

        # If still not available and fallback allowed, use OpenAI fallback
        if self.llm is None and USE_FALLBACK_MODEL and FALLBACK_LLM_MODEL:
            try:
                self.llm = ChatOpenAI(model=FALLBACK_LLM_MODEL, temperature=0)
                logger.info(f"Using OpenAI fallback model: {FALLBACK_LLM_MODEL}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI fallback model: {e}")
                raise

        if self.llm is None:
            raise RAGException("No LLM available: OpenRouter/Gemini candidates failed and no fallback available")
    
    def _make_retrieval_decision(self, query: str) -> AgentDecision:
        """
        Step 1: Agent decides whether retrieval is needed.
        
        Args:
            query: Input question
            
        Returns:
            AgentDecision with decision, confidence, and reasoning
        """
        try:
            decision = needs_retrieval(query, use_llm=True, fallback_to_heuristic=True)
            
            if decision.needs_retrieval:
                logger.info(
                    f"Decision: RETRIEVE (confidence: {decision.confidence:.2f}) - {decision.reasoning}"
                )
            else:
                logger.info(
                    f"Decision: SKIP RETRIEVAL (confidence: {decision.confidence:.2f}) - {decision.reasoning}"
                )
            
            return decision
        except APIException as e:
            logger.warning(f"Retrieval decision failed: {str(e)}, defaulting to RETRIEVE")
            return AgentDecision(
                needs_retrieval=True,
                confidence=0.5,
                reasoning="API error; defaulting to retrieval"
            )
    
    def _retrieve_context(self, query: str) -> Tuple[str, Optional[RetrievalResult]]:
        """
        Step 2-4: Perform hierarchical parent-child retrieval.
        
        Args:
            query: Input question
            
        Returns:
            Tuple of (context_string, retrieval_result)
        """
        try:
            logger.info("[RETRIEVAL] Starting parent-child retrieval...")
            retrieval_result = parent_child_retrieve(
                query,
                self.documents
            )
            
            if not retrieval_result.documents:
                logger.warning("No documents retrieved in parent-child retrieval")
                return "[No context found after retrieval]", retrieval_result
            
            # Extract context from retrieved child chunks
            context = extract_context(retrieval_result.documents)
            logger.info(f"Built context: {len(context)} characters")
            
            return context, retrieval_result
            
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            return f"[Retrieval error: {str(e)}]", None
    
    def answer(
        self,
        query: str,
        return_metadata: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate answer with optional decision tracking.
        
        Full A2-RAG Pipeline:
        1. Decide if retrieval is needed
        2. If yes: Parent-child retrieval with late chunking
        3. Generate answer using context (or direct)
        
        Args:
            query: Input question
            return_metadata: If True, return dict with decision/retrieval info
            
        Returns:
            Answer string, or dict with answer + explainability metadata
            
        Example:
            >>> a2rag = A2RAG(documents)
            >>> result = a2rag.answer(query, return_metadata=True)
            >>> print(f"Needs retrieval: {result['decision'].needs_retrieval}")
            >>> print(f"Answer: {result['answer']}")
        """
        try:
            # Step 1: Make retrieval decision
            decision = self._make_retrieval_decision(query)
            context = None
            retrieval_result = None
            
            # Step 2-4: Optionally retrieve context
            if decision.needs_retrieval:
                context, retrieval_result = self._retrieve_context(query)
                prompt_template = QA_PROMPT_TEMPLATE
            else:
                # Answer directly without retrieval
                logger.info("[GENERATION] Answering directly without retrieval")
                context = "[Direct answer using LLM knowledge]"
                prompt_template = NAIVE_QA_PROMPT_TEMPLATE
            
            # Step 5: Generate answer
            logger.info("[GENERATION] Generating answer with LLM")
            
            if decision.needs_retrieval:
                prompt = prompt_template.format(context=context, query=query)
            else:
                prompt = prompt_template.format(query=query)
            
            try:
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
            
            logger.info(f"[GENERATION] Answer generated ({len(answer)} chars)")
            
            if return_metadata:
                # Extract retrieved doc texts for hit rate calculation
                retrieved_docs = []
                if retrieval_result and retrieval_result.documents:
                    retrieved_docs = [doc.content if hasattr(doc, 'content') else str(doc) for doc in retrieval_result.documents]
                
                # Calculate total API calls:
                # - Decision LLM call: 1 (always made)
                # - Generation LLM call: 1 (always made)
                # - Retrieval queries: num_queries from parent_child_retrieve (only if retrieval was done)
                # Total = 2 (decision + generation) OR 2 + retrieval_result.num_queries if retrieval happened
                total_queries = 2  # Base: decision LLM + generation LLM
                if retrieval_result and retrieval_result.num_queries:
                    total_queries = 2 + retrieval_result.num_queries
                
                return {
                    "answer": answer,
                    "decision": {
                        "needs_retrieval": decision.needs_retrieval,
                        "confidence": decision.confidence,
                        "reasoning": decision.reasoning
                    },
                    "retrieval": {
                        "context_length": len(context) if context else 0,
                        "num_queries": total_queries,  # Total API calls for this query
                        "stage": retrieval_result.retrieval_stage if retrieval_result else "decision_only",
                        "num_documents": len(retrieval_result.documents) if retrieval_result else 0,
                        "documents": retrieved_docs  # Include retrieved documents for hit rate calc
                    },
                    "model": LLM_MODEL
                }
            
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            error_msg = f"Error: {str(e)}"
            
            if return_metadata:
                return {
                    "answer": error_msg,
                    "decision": {"needs_retrieval": None, "confidence": 0, "reasoning": "Error"},
                    "retrieval": {"context_length": 0, "num_queries": 0, "stage": "error", "num_documents": 0},
                    "model": LLM_MODEL
                }
            
            return error_msg
    
    def batch_answer(
        self,
        queries: List[str],
        return_metadata: bool = False
    ) -> List[Union[str, Dict[str, Any]]]:
        """
        Generate answers for multiple queries.
        
        Args:
            queries: List of questions
            return_metadata: If True, return full metadata for each
            
        Returns:
            List of answers (or dicts if return_metadata=True)
        """
        logger.info(f"Processing batch of {len(queries)} queries")
        results = []
        
        for i, query in enumerate(queries, 1):
            try:
                result = self.answer(query, return_metadata=return_metadata)
                results.append(result)
                if i % 10 == 0:
                    logger.info(f"Processed {i}/{len(queries)} queries")
            except Exception as e:
                logger.warning(f"Failed on query {i}: {str(e)}")
                error_result = {
                    "answer": f"Error: {str(e)}",
                    "error": True
                }
                results.append(error_result if return_metadata else f"Error: {str(e)}")
        
        logger.info(f"Batch processing complete: {len(results)} results")
        return results

