"""NVIDIA NIM Two-Stage RAG Engine (Embedding, Cross-Encoder Reranking, and LLM Synthesis)."""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from sec_nim_alpha.config import CONFIG, CHROMA_DIR
from sec_nim_alpha.schemas import (
    ExecutionManifest,
    RAGResponse,
    ComponentStatus,
    ExecutionMode,
)
from sec_nim_alpha.utils import compute_lexical_groundedness, get_logger

logger = get_logger(__name__)


class StableHashEmbeddings(Embeddings):
    """
    Stable deterministic embedding fallback using SHA256.
    Produces consistent vectors across process restarts.
    Only for testing/offline demo - not semantic quality.
    """

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        # Use SHA256 for stable hashing
        hash_obj = hashlib.sha256(text.lower().encode())
        hash_bytes = hash_obj.digest()

        vec = [0.0] * self.dim
        for idx, byte_val in enumerate(hash_bytes):
            slot = idx % self.dim
            vec[slot] += float(byte_val)

        # Normalize
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [round(x / norm, 6) for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@dataclass
class NVIDIANimRAGEngine:
    """Production Two-Stage RAG pipeline using NVIDIA NIM Endpoints with explicit fallback tracking."""

    SYSTEM_PROMPT = (
        "You are a Principal Institutional Equity Analyst. Answer the user's question strictly "
        "using the retrieved SEC 10-K filing passages below. Cite the exact section and chunk_id "
        "in brackets like [Item 7: MD&A | NVDA_abc123_Item7_0] for every factual claim. "
        "Also incorporate the provided FinBERT Management Optimism Index where relevant.\n\n"
        "Quantitative FinBERT Sentiment Context:\n{sentiment_context}\n\n"
        "Retrieved SEC Filing Context:\n{context}"
    )

    api_key: Optional[str] = None
    execution_mode: ExecutionMode = ExecutionMode.DEMO
    nim_enabled: bool = field(default=False, init=False)
    vectorstore: Optional[Chroma] = field(default=None, init=False)
    documents: list[Document] = field(default_factory=list, init=False)
    embeddings: Any = field(default=None, init=False)
    reranker: Any = field(default=None, init=False)
    llm: Any = field(default=None, init=False)
    engine_mode: str = field(default="", init=False)
    _embedding_status: ComponentStatus = field(default=ComponentStatus.SKIPPED, init=False)
    _reranker_status: ComponentStatus = field(default=ComponentStatus.SKIPPED, init=False)
    _generator_status: ComponentStatus = field(default=ComponentStatus.SKIPPED, init=False)

    def __post_init__(self) -> None:
        # Explicitly handle None vs empty string
        if self.api_key is None:
            self.api_key = CONFIG.nvidia_api_key
        self.api_key = self.api_key.strip() if self.api_key else ""
        self.nim_enabled = bool(self.api_key and self.api_key.startswith("nvapi-"))
        self._init_clients()

    def _init_clients(self) -> None:
        """Initializes NVIDIA NIM Embeddings, Reranker, and ChatNVIDIA clients or fallback."""
        if self.nim_enabled and self.execution_mode != ExecutionMode.DEMO:
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
                self._embedding_status = ComponentStatus.SUCCESS
                self._reranker_status = ComponentStatus.SUCCESS
                self._generator_status = ComponentStatus.SUCCESS
                self.engine_mode = f"NVIDIA NIM Cloud ({CONFIG.nim_llm_model})"
                logger.info("Initialized live NVIDIA NIM endpoints successfully.")
                return
            except Exception as exc:
                logger.warning("NVIDIA NIM client init failed (%s). Activating local fallback.", exc)
                self._embedding_status = ComponentStatus.FAILED
                self._reranker_status = ComponentStatus.FAILED
                self._generator_status = ComponentStatus.FAILED

        # Offline fallback
        self.nim_enabled = False
        self.embeddings = StableHashEmbeddings(dim=128)
        self.reranker = None
        self.llm = None
        self._embedding_status = ComponentStatus.OFFLINE_FALLBACK
        self._reranker_status = ComponentStatus.OFFLINE_FALLBACK
        self._generator_status = ComponentStatus.OFFLINE_FALLBACK
        self.engine_mode = "Local Deterministic Fallback (Set NVIDIA_API_KEY for live NIM)"
        logger.info("Running in offline-safe RAG mode.")

    def _get_stable_collection_name(self, ticker: str, accession: str) -> str:
        """Generate stable collection name from ticker and accession."""
        core = f"{ticker}_{accession}_{CONFIG.nim_embed_model}"
        hash_suffix = hashlib.sha256(core.encode()).hexdigest()[:12]
        return f"sec_{ticker.lower()}_{hash_suffix}"

    def index_documents(
        self,
        documents: list[Document],
        collection_name: Optional[str] = None,
        ticker: Optional[str] = None,
        accession: Optional[str] = None,
    ) -> int:
        """Embeds and indexes SEC filing chunks into a persistent Chroma vector store."""
        if not documents:
            raise ValueError("Cannot index an empty document list.")

        self.documents = documents

        # Use stable collection name if ticker and accession provided
        if collection_name is None and ticker and accession:
            collection_name = self._get_stable_collection_name(ticker, accession)
        elif collection_name is None:
            # Fallback with timestamp but warn
            collection_name = f"sec_ephemeral_{int(time.time() * 1000) % 100000}"
            logger.warning("Using ephemeral collection name; provide ticker and accession for stable indexing")

        persist_dir = str(CHROMA_DIR) if CONFIG.chroma_persist else None

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=persist_dir,
        )

        if CONFIG.chroma_persist and hasattr(self.vectorstore, "persist"):
            self.vectorstore.persist()

        logger.info("Indexed %d chunks into ChromaDB collection: %s", len(documents), collection_name)
        return len(documents)

    def retrieve_and_rerank(self, query: str) -> tuple[list[Document], float, dict[str, Any]]:
        """
        Executes Stage-1 vector search followed by Stage-2 NVIDIA NIM cross-encoder reranking.
        Returns (top_docs, latency_ms, retrieval_metadata).
        """
        if self.vectorstore is None:
            raise RuntimeError("Vector store not initialized. Call index_documents() first.")

        t0 = time.perf_counter()
        k_initial = min(CONFIG.top_k_initial, len(self.documents))
        candidates = self.vectorstore.similarity_search(query, k=max(1, k_initial))
        retrieval_count = len(candidates)

        rerank_metadata = {
            "reranker_status": ComponentStatus.SKIPPED.value,
            "rerank_count": 0,
        }

        if self.nim_enabled and self.reranker is not None and len(candidates) > 1:
            try:
                reranked = list(self.reranker.compress_documents(documents=candidates, query=query))
                top_docs = reranked[:CONFIG.top_k_rerank]
                rerank_metadata["reranker_status"] = ComponentStatus.SUCCESS.value
                rerank_metadata["rerank_count"] = len(top_docs)
                self._reranker_status = ComponentStatus.SUCCESS
            except Exception as exc:
                logger.warning("NVIDIA Reranker call failed (%s); using Stage-1 vector rank.", exc)
                top_docs = candidates[:CONFIG.top_k_rerank]
                rerank_metadata["reranker_status"] = ComponentStatus.DEGRADED.value
                rerank_metadata["rerank_count"] = len(top_docs)
                rerank_metadata["rerank_error"] = str(exc)
                self._reranker_status = ComponentStatus.DEGRADED
        else:
            # Lexical overlap fallback reranking
            query_terms = set(query.lower().split())
            scored = sorted(
                candidates,
                key=lambda d: sum(1 for w in query_terms if w in d.page_content.lower()),
                reverse=True,
            )
            top_docs = scored[:CONFIG.top_k_rerank]
            rerank_metadata["reranker_status"] = ComponentStatus.OFFLINE_FALLBACK.value
            rerank_metadata["rerank_count"] = len(top_docs)
            self._reranker_status = ComponentStatus.OFFLINE_FALLBACK

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return top_docs, latency_ms, rerank_metadata

    def answer_question(
        self,
        query: str,
        sentiment_summary: Optional[dict[str, Any]] = None,
        filing_accession: Optional[str] = None,
    ) -> RAGResponse:
        """Runs full Two-Stage RAG + synthesis and returns response with execution manifest."""
        top_docs, retrieval_ms, rerank_meta = self.retrieve_and_rerank(query)

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
        generator_status = ComponentStatus.SKIPPED

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
                generator_status = ComponentStatus.SUCCESS
                self._generator_status = ComponentStatus.SUCCESS
            except Exception as exc:
                logger.warning("NIM LLM synthesis error (%s). Falling back to extractive synthesis.", exc)
                answer_text = self._extractive_fallback(query, top_docs, sentiment_str)
                generator_status = ComponentStatus.DEGRADED
                self._generator_status = ComponentStatus.DEGRADED
        else:
            answer_text = self._extractive_fallback(query, top_docs, sentiment_str)
            generator_status = ComponentStatus.OFFLINE_FALLBACK
            self._generator_status = ComponentStatus.OFFLINE_FALLBACK

        gen_ms = round((time.perf_counter() - t_gen) * 1000.0, 2)
        raw_texts = [d.page_content for d in top_docs]
        groundedness = compute_lexical_groundedness(answer_text, raw_texts)

        citations = [
            {
                "chunk_id": d.metadata.get("chunk_id", "N/A"),
                "section": d.metadata.get("section", "SEC Filing"),
                "filing_date": d.metadata.get("filing_date", "N/A"),
                "accession": d.metadata.get("accession_number", "N/A"),
                "snippet": d.page_content[:280] + ("..." if len(d.page_content) > 280 else ""),
            }
            for d in top_docs
        ]

        # Build execution manifest
        manifest = ExecutionManifest(
            data_mode=self.execution_mode,
            filing_accession=filing_accession,
            index_version=CONFIG.nim_embed_model,
            embedding_model=CONFIG.nim_embed_model,
            embedding_status=self._embedding_status,
            reranker_model=CONFIG.nim_rerank_model,
            reranker_status=ComponentStatus(rerank_meta["reranker_status"]),
            generator_model=CONFIG.nim_llm_model,
            generator_status=generator_status,
            sentiment_backend=sentiment_summary.get("backend", "unknown") if sentiment_summary else "unknown",
            sentiment_status=ComponentStatus.SUCCESS,
            market_data_status=ComponentStatus.SUCCESS,
            warnings=tuple(),
        )

        # Update engine_mode to reflect actual status for this request
        if generator_status == ComponentStatus.SUCCESS:
            actual_mode = f"NVIDIA NIM ({CONFIG.nim_llm_model})"
        elif generator_status == ComponentStatus.DEGRADED:
            actual_mode = f"NVIDIA NIM (degraded: extractive fallback)"
        else:
            actual_mode = "Local Deterministic Fallback"

        return RAGResponse(
            query=query,
            answer=answer_text,
            citations=citations,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=gen_ms,
            groundedness_score=groundedness,
            engine_mode=actual_mode,
            execution_manifest=manifest,
            retrieval_count=rerank_meta.get("rerank_count", 0) + len(top_docs),
            rerank_count=rerank_meta.get("rerank_count", 0),
            insufficient_evidence=len(top_docs) == 0,
        )

    @staticmethod
    def _extractive_fallback(query: str, docs: list[Document], sentiment_str: str) -> str:
        """Synthesizes a deterministic, grounded analyst briefing when running without an API key."""
        if not docs:
            return (
                f"### Grounded SEC Filing Analysis\n"
                f"**Query:** *{query}*\n\n"
                f"**Quantitative Tone Signal:** {sentiment_str}\n\n"
                f"**Key Filing Evidence:** No relevant passages retrieved."
            )

        bullet_points = []
        for doc in docs:
            sec = doc.metadata.get("section", "SEC Section")
            cid = doc.metadata.get("chunk_id", "chunk_0")
            excerpt = doc.page_content.strip()
            bullet_points.append(f"- **[{sec} | {cid}]**: {excerpt}")

        joined_bullets = "\n".join(bullet_points)
        return (
            f"### Grounded SEC Filing Analysis (Extractive Synthesis)\n"
            f"**Query:** *{query}*\n\n"
            f"**Quantitative Tone Signal:** {sentiment_str}\n\n"
            f"**Key Filing Evidence (Retrieved Passages):**\n{joined_bullets}"
        )