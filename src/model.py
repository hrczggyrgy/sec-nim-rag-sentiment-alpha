"""NVIDIA NIM Two-Stage RAG Engine (Embedding, Cross-Encoder Reranking, and Llama-3.3-70B Synthesis)."""

import time
from dataclasses import dataclass
from typing import Any, Optional
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from src.config import CONFIG
from src.utils import compute_lexical_groundedness, get_logger

logger = get_logger(__name__)


class DeterministicHashEmbeddings(Embeddings):
    """Lightweight deterministic embedding fallback for offline CI/CD testing when no API key is set."""

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for idx, token in enumerate(text.lower().split()):
            slot = (hash(token) + idx) % self.dim
            vec[slot] += 1.0
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [round(x / norm, 6) for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@dataclass
class RAGResponse:
    """Structured response payload including citations and MLOps observability telemetry."""

    query: str
    answer: str
    citations: list[dict[str, Any]]
    retrieval_latency_ms: float
    generation_latency_ms: float
    groundedness_score: float
    engine_mode: str


class NVIDIANimRAGEngine:
    """Production Two-Stage RAG pipeline using NVIDIA NIM Endpoints with automatic offline fallback."""

    SYSTEM_PROMPT = (
        "You are a Principal Institutional Equity Analyst. Answer the user's question strictly "
        "using the retrieved SEC 10-K filing passages below. Cite the exact section and chunk_id "
        "in brackets like [Item 7: MD&A | NVDA_Item7:_0] for every factual claim. "
        "Also incorporate the provided FinBERT Management Optimism Index where relevant.\n\n"
        "Quantitative FinBERT Sentiment Context:\n{sentiment_context}\n\n"
        "Retrieved SEC Filing Context:\n{context}"
    )

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = (api_key or CONFIG.nvidia_api_key).strip()
        self.nim_enabled = bool(self.api_key and self.api_key.startswith("nvapi-"))
        self.vectorstore: Optional[Chroma] = None
        self.documents: list[Document] = []
        self._init_clients()

    def _init_clients(self) -> None:
        """Initializes NVIDIA NIM Embeddings, Reranker, and ChatNVIDIA clients or fallback."""
        if self.nim_enabled:
            try:
                from langchain_nvidia_ai_endpoints import (
                    ChatNVIDIA,
                    NVIDIAEmbeddings,
                    NVIDIARerank,
                )

                self.embeddings = NVIDIAEmbeddings(
                    model=CONFIG.nim_embed_model,
                    nvidia_api_key=self.api_key,
                    truncate="END",
                )
                self.reranker = NVIDIARerank(
                    model=CONFIG.nim_rerank_model,
                    nvidia_api_key=self.api_key,
                    top_n=CONFIG.top_k_rerank,
                )
                self.llm = ChatNVIDIA(
                    model=CONFIG.nim_llm_model,
                    nvidia_api_key=self.api_key,
                    temperature=CONFIG.llm_temperature,
                    max_tokens=CONFIG.llm_max_tokens,
                )
                self.engine_mode = f"NVIDIA NIM Cloud ({CONFIG.nim_llm_model})"
                logger.info("Initialized live NVIDIA NIM endpoints successfully.")
                return
            except Exception as exc:
                logger.warning("NVIDIA NIM client init failed (%s). Activating local fallback.", exc)

        self.nim_enabled = False
        self.embeddings = DeterministicHashEmbeddings(dim=128)
        self.reranker = None
        self.llm = None
        self.engine_mode = "Local Deterministic Fallback (Set NVIDIA_API_KEY for live NIM)"
        logger.info("Running in offline-safe RAG mode.")

    def index_documents(self, documents: list[Document], collection_name: str = "sec_10k_store") -> int:
        """Embeds and indexes SEC filing chunks into an in-memory Chroma vector store."""
        if not documents:
            raise ValueError("Cannot index an empty document list.")
        self.documents = documents
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=f"{collection_name}_{int(time.time() * 1000) % 100000}",
        )
        logger.info("Indexed %d chunks into ChromaDB collection.", len(documents))
        return len(documents)

    def retrieve_and_rerank(self, query: str) -> tuple[list[Document], float]:
        """Executes Stage-1 vector search followed by Stage-2 NVIDIA NIM cross-encoder reranking."""
        if self.vectorstore is None:
            raise RuntimeError("Vector store not initialized. Call index_documents() first.")

        t0 = time.perf_counter()
        k_initial = min(CONFIG.top_k_initial, len(self.documents))
        candidates = self.vectorstore.similarity_search(query, k=max(1, k_initial))

        if self.nim_enabled and self.reranker is not None and len(candidates) > 1:
            try:
                reranked = list(self.reranker.compress_documents(documents=candidates, query=query))
                top_docs = reranked[: CONFIG.top_k_rerank]
            except Exception as exc:
                logger.warning("NVIDIA Reranker call failed (%s); using Stage-1 vector rank.", exc)
                top_docs = candidates[: CONFIG.top_k_rerank]
        else:
            query_terms = set(query.lower().split())
            scored = sorted(
                candidates,
                key=lambda d: sum(1 for w in query_terms if w in d.page_content.lower()),
                reverse=True,
            )
            top_docs = scored[: CONFIG.top_k_rerank]

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return top_docs, latency_ms

    def answer_question(
        self,
        query: str,
        sentiment_summary: Optional[dict[str, Any]] = None,
    ) -> RAGResponse:
        """Runs full Two-Stage RAG + synthesis and logs MLOps groundedness telemetry."""
        top_docs, retrieval_ms = self.retrieve_and_rerank(query)

        context_blocks = [
            f"[{d.metadata.get('section')} | {d.metadata.get('chunk_id')}]\n{d.page_content}"
            for d in top_docs
        ]
        context_str = "\n\n".join(context_blocks)
        sentiment_str = (
            f"Management Optimism Index (MOI): {sentiment_summary.get('optimism_index', 0.0):+.3f} "
            f"(Positive: {sentiment_summary.get('positive_ratio', 0.0)*100:.1f}%, "
            f"Negative: {sentiment_summary.get('negative_ratio', 0.0)*100:.1f}%)"
            if sentiment_summary
            else "No quantitative sentiment summary provided."
        )

        t_gen = time.perf_counter()
        if self.nim_enabled and self.llm is not None:
            try:
                prompt = ChatPromptTemplate.from_messages(
                    [("system", self.SYSTEM_PROMPT), ("human", "{question}")]
                )
                chain = prompt | self.llm
                result = chain.invoke(
                    {
                        "sentiment_context": sentiment_str,
                        "context": context_str,
                        "question": query,
                    }
                )
                answer_text = str(result.content)
            except Exception as exc:
                logger.warning("NIM LLM synthesis error (%s). Falling back to extractive synthesis.", exc)
                answer_text = self._extractive_fallback(query, top_docs, sentiment_str)
        else:
            answer_text = self._extractive_fallback(query, top_docs, sentiment_str)

        gen_ms = round((time.perf_counter() - t_gen) * 1000.0, 2)
        raw_texts = [d.page_content for d in top_docs]
        groundedness = compute_lexical_groundedness(answer_text, raw_texts)

        citations = [
            {
                "chunk_id": d.metadata.get("chunk_id", "N/A"),
                "section": d.metadata.get("section", "SEC Filing"),
                "filing_date": d.metadata.get("filing_date", "N/A"),
                "snippet": d.page_content[:280] + ("..." if len(d.page_content) > 280 else ""),
            }
            for d in top_docs
        ]

        return RAGResponse(
            query=query,
            answer=answer_text,
            citations=citations,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=gen_ms,
            groundedness_score=groundedness,
            engine_mode=self.engine_mode,
        )

    @staticmethod
    def _extractive_fallback(query: str, docs: list[Document], sentiment_str: str) -> str:
        """Synthesizes a deterministic, grounded analyst briefing when running without an API key."""
        bullet_points = []
        for doc in docs:
            sec = doc.metadata.get("section", "SEC Section")
            cid = doc.metadata.get("chunk_id", "chunk_0")
            excerpt = doc.page_content.strip()
            bullet_points.append(f"- **[{sec} | {cid}]**: {excerpt}")

        joined_bullets = "\n".join(bullet_points)
        return (
            f"### Grounded SEC Filing Analysis\n"
            f"**Query:** *{query}*\n\n"
            f"**Quantitative Tone Signal:** {sentiment_str}\n\n"
            f"**Key Filing Evidence (Two-Stage Retrieved Passages):**\n{joined_bullets}"
        )