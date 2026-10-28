# Technical Review and Improvement Plan

## SEC NIM RAG Sentiment Alpha

- Repository: `hrczggyrgy/sec-nim-rag-sentiment-alpha`
- Reviewed commit: `68fff3ca135f3edb9827d9ffeac026e8506eb32d`
- Review date: October 28, 2026
- Method: authenticated, read-only GitHub inspection and supporting public technical documentation.
- Scope: repository root, `README.md`, `src/config.py`, `src/data_loader.py`, `src/features.py`, `src/model.py`, `src/utils.py`, `app/main.py`, `tests/test_pipeline.py`, `pyproject.toml`, and `Dockerfile`.
- Limitations: static review only; no tests, endpoint probes, GPU benchmarks, dependency resolution, or backtests were executed. `requirements.txt`, `uv.lock`, notebooks, and `blueprint.md` were not inspected. Endpoint availability and commit-message performance claims remain unverified.
- Evidence notation: implementation findings identify the inspected file and function at the pinned commit. External citations support technical context and recommendations, not claims that repository code was executed.

## 1. Executive Assessment

The repository is a useful research **prototype**, not yet a production-grade alpha engine. It connects filing ingestion, a two-stage retrieval interface, sentiment scoring, market metrics, and an analyst dashboard. Its principal blockers are data provenance, misleading fallback behavior, point-in-time correctness, and insufficient evaluation—not a lack of larger models or distributed infrastructure.

### Recommended order of investment

1. High: prevent synthetic text, generated prices, and simulated history from entering research outputs unnoticed.
2. High: establish filing-level provenance, public-availability timestamps, reproducible indices, and honest component-level execution status.
3. High: replace self-confirming quality proxies with retrieval, citation, sentiment, and economic evaluation.
4. Medium: separate offline ingestion/features from online querying and tune NIM using measured workload characteristics.
5. Medium/Low: expand infrastructure only after corpus size, concurrency, and service objectives justify it.

### Existing strengths

- Separate loader, feature, retrieval, configuration, and utility modules provide a workable starting point.
- Dataclasses make core payloads reasonably explicit.
- Reranker and generation failures do not necessarily prevent a demonstration from running.
- Documents include accession and section metadata, although identifiers and citations need strengthening.
- A lockfile and an initial test suite exist; their existence does not establish reproducibility or comprehensive coverage.

## 2. Verified Findings and Immediate Remediation

| Priority | Finding and implementation evidence | Impact | Required action |
|---|---|---|---|
| High | `SECDataLoader.fetch_market_metrics()` returns seeded random returns and volatility after any exception, without a separate market-data provenance flag. | A live filing can receive fabricated outcome labels while `is_synthetic_fallback` remains false. | Return nullable metrics with explicit status/error; permit generated labels only in isolated demo fixtures. |
| High | `load_historical_panel()` generates all four rows randomly; `run_full_pipeline()` replaces only the final row's metrics. | The trend chart mixes three simulated historical observations with one current observation. | Remove the chart from research mode until actual accession-level historical observations exist; watermark demo mode. |
| High | Unsupported-ticker fallback uses `SYNTHETIC_SEC_CORPUS.get(ticker, ...['NVDA'])` but preserves the requested ticker. | Another issuer's text and accession can be attributed to the requested security. | Reject unsupported demo tickers or preserve the actual fixture issuer explicitly. |
| High | Filing cache uses `{ticker}_10k.json`, ignores accession/version, and can persist synthetic records. | Stale or synthetic data can survive network recovery and contaminate later runs. | Isolate demo/live namespaces; cache immutable filings by accession and track freshness separately. |
| High | `fetch_market_metrics()` anchors on filing date and accepts as few as four bars despite a seven-return objective. | Event alignment is ambiguous; incomplete horizons receive apparently complete labels. | Use public availability plus processing latency and an explicit execution rule; require the full horizon or mark labels pending. |
| High | `DeterministicHashEmbeddings._embed()` uses Python `hash()` and token position. | Vectors are not stable across ordinary process restarts and are weak semantic retrieval substitutes. | Restrict to tests and use a stable digest; use a versioned semantic model for real local retrieval. |
| High | Reranker/LLM call failures do not update `engine_mode` or `nim_enabled`. | The UI can still say NVIDIA NIM when generation is extractive or reranking failed. | Report actual provider/model/status per stage, per request. |
| High | `compute_lexical_groundedness()` measures unique-token substring overlap; fallback output copies passages. | Copying can score well without answering correctly; contradictions and unsupported relations can pass. | Rename to lexical overlap and add claim-level support, contradiction, and citation checks. |
| High | Sentiment examines only the first 40 qualifying sentences, MD&A before risk factors. | Coverage depends on document order; later risks may be absent from the score. | Score the eligible corpus or stratify by section with explicit coverage weights. |
| Medium | Section extraction falls back to splitting the first 35,000 characters in half; live sections are capped at 30,000 characters each. | Chunks can carry incorrect Item labels or omit material text. | Parse actual boundaries and retain full raw sources; mark section-extraction failures explicitly. |
| Medium | `index_documents()` creates an ephemeral Chroma collection using a millisecond suffix modulo 100,000. | No deliberate durable index lifecycle; names can collide and repeated runs can leave unused collections. | Use stable versioned collection identities, explicit persistence, idempotent upserts, and cleanup. |
| Medium | `api_key or CONFIG.nvidia_api_key` ignores an explicitly empty key. | Tests and UI cannot reliably force offline mode when an environment key exists. | Distinguish `None` from `''`; inject an explicit execution policy. |
| Medium | README/UI model descriptions differ from `src/config.py`. | Operators cannot reliably infer the active stack. | Generate displayed model metadata from runtime configuration and a tested capability registry. |

The active configuration names `nvidia/nemotron-3.5-embed-8b`, `nvidia/llama-3.1-nemotron-ultra-253b-v1`, and `nvidia/llama-3.2-nv-rerankqa-1b-v2`. These are observed identifiers, not a verification of endpoint availability, deprecation status, or successful inference.

## 3. Architecture and Performance

### 3.1 High priority: separate research data creation from interactive querying

Keep a modular monolith initially, but introduce separate execution paths:

```text
SEC discovery -> rate-limited fetch -> immutable raw filing store
                                      |
                               parse / validate
                                      |
                      accession-keyed document/chunk store
                         |                         |
                 batched embeddings       sentiment extraction
                         |                         |
                 versioned search index   point-in-time feature table
                         |                         |
Streamlit -> query API -> filtered retrieval -> reranking -> cited answer

Market data -> independently versioned future labels -> offline evaluation
```

- Move ingestion, embedding, and sentiment computation into background jobs; querying should not rebuild the corpus.
- Keep Streamlit as a presentation layer. Introduce FastAPI when independent clients, access control, or deployment scaling justify it.
- Use a task queue such as Celery/RQ for a small deployment; use Dagster/Prefect when scheduled backfills and lineage become operational needs. Do not adopt all three.
- Store raw filings in versioned object storage and metadata in PostgreSQL. Use an accession/content-hash idempotency key and atomic job-state transitions.
- Separate future outcomes from the feature schema. Query-time analysis must not receive post-event labels accidentally.

### 3.2 High priority: make failure modes explicit

Define separate `demo`, `research`, and `production` policies:

- Demo: synthetic fixtures and lightweight substitutes allowed; every output visibly labeled.
- Research: missing market data remains missing; synthetic observations excluded by construction; model changes create distinct feature versions.
- Production: typed failures, explicit degraded responses, bounded retries, and abstention when evidence is inadequate.

Return an execution manifest containing:

```json
{
  "data_mode": "live",
  "filing_accession": "...",
  "index_version": "...",
  "embedding_model": "...",
  "reranker_status": "skipped_or_failed_or_success",
  "generator_status": "extractive_or_llm_or_failed",
  "sentiment_backend": "finbert_or_lexicon",
  "warnings": []
}
```

Do not present extractive synthesis as a successful LLM call or a lexicon score as FinBERT inference.

### 3.3 Medium priority: optimize NVIDIA NIM utilization

The implementation calls NVIDIA-hosted endpoints; it does not deploy local NIM containers. Maintain that distinction when attributing latency, cost, or GPU utilization. NVIDIA documents separate embedding, reranking, generation, and deployment scaling considerations. [web:4][web:5]

- Configure provider, base URL, model/revision, timeout, token limit, and retry policy independently for each component.
- Probe capabilities at startup with small contract requests. Client construction alone is not a service-health check.
- Use pooled async clients, bounded concurrency, request deadlines, retry budgets, jittered backoff for transient failures, and circuit breakers. Surface authorization and invalid-model errors rather than retrying indefinitely.
- Batch offline document embeddings by token budget. Keep query embedding requests on a latency-oriented queue so bulk ingestion cannot starve interactive traffic.
- Evaluate a smaller generator against the configured 253B model on the same grounded-answer benchmark; a larger model is not an automatic quality or cost win.
- Cache embeddings by content hash plus model/preprocessing version; cache retrieval by query, filters, and index version. Include corpus cutoff and sentiment version in answer-cache keys.
- For self-hosting, benchmark supported NIM profiles and batching settings; persist model caches, separate workload queues, and monitor GPU memory, utilization, queue wait, and tokens/second. Consider MIG only where the selected GPU/model profile supports it.
- Retain hosted inference until measured volume and control requirements justify operating GPUs. Kubernetes is an operational choice, not an initial prerequisite.

Embedding plus reranking is a supported retrieval pattern, but its quality/latency trade-off must be measured on this corpus rather than inherited from a vendor benchmark. [web:6][web:8]

### 3.4 Performance acceptance criteria

- Instrument ingestion, query embedding, search, reranking, prompt construction, generation, and total latency separately.
- Load-test warm/cold paths and expected concurrency; report p50/p95/p99, time to first token, errors, and cost per completed task.
- Establish measured baseline budgets before setting absolute latency targets. README mock-display values are not benchmark evidence.
- Require no repeated document embedding during a warm query against an unchanged index.

## 4. RAG and Information Retrieval

### 4.1 High priority: preserve SEC structure and provenance

- Replace arbitrary character slicing with section-aware HTML/XBRL extraction. Preserve headings, paragraph boundaries, list structure, tables, units, and footnotes.
- Validate that Item 1A and Item 7 actually match their labels. Store extraction status and parser version; do not silently relabel generic filing text.
- Use token-aware child chunks as an initial experiment: approximately 300–600 tokens, with 10–15% overlap, constrained by the actual embedding and reranker limits. These are tuning starting points, not proven optima.
- Retrieve child passages and expand to their parent paragraphs/subsections when necessary. Avoid splitting financial tables across chunks without repeating headers and units.
- Construct stable identifiers from accession, section, source offsets, content hash, and preprocessing version. Current IDs omit accession and can repeat across filing years.
- Add CIK, accession, form, period end, public-availability time, ingestion time, source URL, offsets, and document hash to metadata and returned citations.

### 4.2 High priority: improve retrieval before upgrading generation

- Add lexical retrieval, such as BM25, alongside dense search; combine rankings with reciprocal rank fusion.
- Apply issuer, form, section, and as-of filters before candidate selection. Cross-company retrieval must be an explicit mode.
- Test a larger candidate set, initially 30–100, followed by reranking to 5–10 passages; tune against recall, redundancy, token budget, and latency.
- Deduplicate overlapping passages and optionally diversify by subsection.
- Add comparative retrieval for the current and prior filing, restricted to documents available at the analysis cutoff.
- Return an insufficient-evidence answer when candidate relevance is inadequate. Calibrate thresholds on held-out queries.

### 4.3 Medium priority: select embeddings through a reproducible bake-off

Compare the configured NVIDIA embedding endpoint, a supported lighter NVIDIA alternative, and a local baseline such as the configured but currently unused `sentence-transformers/all-MiniLM-L6-v2`.

- Evaluate SEC-specific relevance, truncation, input-type handling, normalization, dimension, latency, throughput, cost, and deployment constraints.
- Verify the vector-store distance metric explicitly; the README's cosine-search description is not established by the inspected constructor.
- Pin model revisions where possible and record endpoint metadata where immutable revisions are unavailable.
- Never mix incompatible embedding models in one collection. Model or dimensionality changes require a separately built and validated index before cutover. NVIDIA's reference guidance likewise requires consistent ingestion/query embeddings. [web:12]
- Keep hash vectors strictly in unit tests; stable hashing repairs reproducibility, not semantic quality.

### 4.4 Medium priority: vector-store decision

| Deployment need | Suggested starting point | Decision criterion |
|---|---|---|
| Small local demonstration | Explicitly persisted Chroma | Simple lifecycle and modest corpus/concurrency |
| Research service already using PostgreSQL | PostgreSQL + pgvector, with lexical search | Operational simplicity, provenance joins, backup/restore |
| Dedicated retrieval service | Evaluate Qdrant or Milvus | Measured filtered-search throughput, scale, tenancy, operational capacity |

Do not migrate solely because a database is popular. Benchmark with expected corpus size, metadata selectivity, concurrent queries, update volume, recall, and recovery requirements. Keep raw text and lineage recoverable independently of the vector index.

### 4.5 High priority: retrieval and grounded-answer evaluation

Create a human-reviewed benchmark spanning issuers, sectors, years, and question types:

- Exact facts, numerical comparisons, multi-passage questions, negative questions, and unanswerable questions.
- Gold passage sets and source offsets, not just answer strings.
- Recall@k, nDCG@k, MRR, citation precision/coverage, numerical accuracy, unsupported-claim rate, and abstention quality.
- A temporal holdout and an issuer holdout; prevent near-duplicate filing text from crossing evaluation boundaries.
- An ablation matrix: dense only; hybrid; hybrid plus reranker; alternate chunks; alternate embeddings; alternate generators.

Treat retrieved documents as untrusted data, not instructions. Keep filing content out of privileged instruction text, test prompt-injection passages, validate returned citation IDs against the retrieved set, and verify claim support rather than merely citation presence.

## 5. Sentiment Analysis and Alpha Modeling

### 5.1 High priority: repair sentiment coverage and backend semantics

Observed behavior in `src/features.py`:

- Regex sentence splitting and a 45-character threshold select eligible sentences.
- Only the first 40 sentences are analyzed, ordered MD&A then risk factors.
- The model returns only each sentence's winning label/confidence; the score aggregates confidence for winning positive/negative labels.
- The fallback is a small hardcoded vocabulary, not an implemented full Loughran–McDonald dictionary.
- Missing positive/negative examples are replaced with arbitrary sentences for the “top bullish/bearish” fields.

Required improvements:

- Use a finance-tested sentence segmenter and evaluate abbreviations, decimals, headings, and tables.
- Score all eligible sentences offline, or use documented stratified sampling with inclusion weights and section coverage.
- Obtain all class probabilities. An interpretable baseline is:

\[
s_i=P_i(\mathrm{positive})-P_i(\mathrm{negative}),\qquad
S_{d,k}=\frac{\sum_{i\in(d,k)}w_i s_i}{\sum_{i\in(d,k)}w_i}.
\]

Here \(d\) identifies the filing, \(k\) the section, and \(w_i\) an explicit, versioned weighting rule. This is a proposed replacement, not the current score.

- Keep MD&A optimism, risk negativity, uncertainty, litigation, liquidity, and forward-looking tone as separate features.
- Batch inference, configure device placement, use inference mode, and pin tokenizer/model artifacts. Handle overlong sentences explicitly rather than silently losing decisive trailing clauses.
- Return null when no valid sentiment evidence exists. Return no bullish/bearish example when that class is absent.
- Record backend/version and prevent neural and lexicon scores from sharing an unlabeled time series.

FinBERT provides positive, negative, and neutral outputs and was fine-tuned using Financial PhraseBank; that establishes a useful baseline, not validated accuracy on these SEC sections. Review model and dataset licensing before commercial use; the model card identifies separate dataset restrictions. [web:19][web:20]

### 5.2 High priority: validate NLP robustness

Build an expert-labeled SEC sentence set with issuer/temporal separation. Include negation, conditional risks, comparisons, quoted third-party statements, boilerplate, and numerical changes such as increasing losses versus increasing revenue.

Measure macro-F1, per-class precision/recall, calibration error, Brier score, and section/sector breakdowns. Calibrate probabilities on validation data only. Do not treat a softmax confidence or lexicon's invented confidence as a proven reliability estimate.

Start with full-probability FinBERT and a properly versioned lexicon baseline. Consider LoRA/fine-tuning or NIM-based structured extraction only if measured errors justify the additional complexity. Separate extraction accuracy from any downstream trading result.

### 5.3 High priority: establish point-in-time event data

The current pipeline fetches the latest 10-K; it does not construct a historical accession-level training panel. Build that panel before claiming alpha.

- Preserve CIK/security mapping histories, amendments, original raw files, and each source's observation timestamps.
- Store EDGAR acceptance, observed public dissemination, and first successful ingestion separately. Define the earliest usable event time conservatively; acceptance alone is not always public availability.
- Define `signal_available_at` as no earlier than public availability and actual processing completion. Use the next executable price under a documented venue/session rule.
- Keep UTC timestamps plus an explicit exchange calendar/timezone. Test after-hours releases, weekends, holidays, and daylight-saving changes.
- Freeze contemporaneous market-data snapshots and corporate-action handling; use a point-in-time universe including delisted securities where available.
- Require complete future windows before assigning labels; partial windows remain pending.

SEC documentation distinguishes filing dates and dissemination behavior, including submissions disseminated on the following business day. Date-only alignment is therefore insufficient for an event-driven strategy. [web:1][web:2]

### 5.4 Medium priority: turn tone into testable factor hypotheses

Sentiment is not automatically alpha. Post-filing volatility is a risk outcome, not directional excess return.

Candidate features:

- Section-specific tone changes against the issuer's prior comparable filing.
- New negative/uncertainty language relative to retained boilerplate.
- Changes in forward-looking language and confidence.
- Tone residuals after accounting for earnings/revenue surprises and other contemporaneously available fundamentals.

For a prior-comparable-filing feature:

\[
\Delta S_{j,t}=S_{j,t}-S_{j,t^-}.
\]

Estimate a residualized cross-sectional score using only information available at time \(t\):

\[
a_{j,t}=\operatorname{zscore}_t\!\left(\Delta S_{j,t}-\widehat{E}[\Delta S_{j,t}\mid X_{j,t}]\right).
\]

The conditioning model, winsorization, and scaling must be fit without future observations. Do not pool asynchronously released annual filings as though they were simultaneous; define signal age, carry-forward, and decay rules.

### 5.5 High priority: economic validation and portfolio controls

- Begin with linear/rank-based baselines; compare against sector, beta, size, value, momentum, and volatility exposures.
- Use anchored/rolling walk-forward evaluation. Purge overlapping forward-label intervals and apply embargoes where needed; keep a final untouched period.
- Track every model, prompt, feature, horizon, and hyperparameter trial. Report multiple-testing-aware statistics such as Deflated Sharpe Ratio and assess backtest overfitting. [web:18]
- Evaluate rank IC, IC stability, net long-short returns, turnover, drawdown, transaction/borrow costs, liquidity, and capacity.
- Use block/time-clustered uncertainty estimates rather than assuming all filing observations are independent.
- Run ablations against lexicon-only and non-text baselines to test incremental information.
- Add concentration, sector exposure, leverage, liquidity, position-age, and stale-signal limits before paper trading.

A small illustrative correlation or a short realized-volatility window is not evidence of an investable strategy. Backtest success, if later demonstrated, still does not establish live profitability.

## 6. Code Quality, Security, and MLOps

### 6.1 High priority: packaging and boundaries

Recommended incremental layout:

```text
src/sec_nim_alpha/
  config.py
  schemas.py
  ingestion/
  parsing/
  retrieval/
  sentiment/
  research/
  serving/
  observability/
tests/
  unit/
  integration/
  evaluation/
  fixtures/
```

- Replace `from src...` imports and UI `sys.path` mutation with an installable package.
- Fix package discovery: `where=['src']` combined with `include=['src*']` does not describe the current flat module layout correctly. Add wheel-build and installed-import smoke tests.
- Inject filing, market, embedding, reranking, and generation clients through small protocols. Avoid a heavyweight orchestration rewrite.
- Use validated settings, explicit execution modes, and configuration loading without import-time directory creation.
- Keep future labels out of serving DTOs and feature tables.
- Remove tracked build metadata such as `src/*.egg-info`; generate it during packaging.

### 6.2 High priority: tests that can actually fail

The inspected suite contains four tests. Its fixture may call EDGAR and Yahoo Finance, it disables FinBERT for feature tests, and it does not establish live NIM correctness. An environment API key can also defeat the intended offline engine setting.

Add:

- Network-disabled unit tests using local filing fixtures and injected clients.
- Temporary cache/index paths; no shared production state.
- Subprocess tests for embedding determinism and explicit empty-key behavior.
- Provenance tests: unsupported issuer, synthetic cache, real filing plus missing prices, simulated history, incomplete labels.
- Parser tests for actual section boundaries, malformed HTML, tables, and long sections.
- Event-time tests covering same-day premarket/after-hours and holiday/weekend cases.
- Fault tests for 429, timeout, authentication failure, malformed response, reranker failure, LLM failure, and empty retrieval.
- Grounding tests with contradictory numbers, negation, swapped issuers, and fake citation IDs.
- Separately marked, scheduled live contracts; do not run paid endpoints for every pull request.

Use coverage to identify untested critical paths rather than treating an arbitrary percentage as proof of correctness.

### 6.3 Medium priority: CI/CD and supply chain

No `.github` directory appeared in the inspected root listing, so no repository GitHub Actions workflows were visible there.

Suggested PR pipeline:

1. Ruff formatting/lint, mypy, and unit tests without network or secrets.
2. Build wheel; install in a clean environment; run import and CLI/API smoke tests.
3. Run local integration tests and the fixed retrieval/sentiment evaluation suite.
4. Scan secrets and dependencies, build an SBOM, and scan the image with tools such as Gitleaks, pip-audit, and Trivy.
5. Publish an immutable image only after protected-branch checks pass.

Use `pyproject.toml` plus `uv.lock` as the dependency authority, with frozen installs. The current Dockerfile installs `requirements.txt`; verify or generate that file from the lock rather than maintaining unrelated specifications. Move pytest into development dependencies. Pin deployment images by digest and record model/index/evaluation versions in release metadata.

### 6.4 Medium priority: containers and runtime security

Observed Dockerfile concerns: root execution, retained build tools, broad `COPY . .`, no healthcheck, no lockfile-based install, and no visible `.dockerignore` in the root listing.

- Use a multi-stage build and non-root runtime user.
- Add `.dockerignore` exclusions for `.env`, caches, local datasets, virtual environments, credentials, and Git metadata. Broad copying creates an exposure path; this review did not establish that secrets were actually committed or baked into an image.
- Separate UI/API images from self-hosted GPU NIM services.
- Add health/readiness checks, resource limits, graceful shutdown, and persistent volumes only where needed.
- Redact keys and sensitive prompts from logs. Review `st.cache_resource` sharing and credential lifetimes before multi-user deployment; do not share user-bound clients across tenants unintentionally.
- Add authentication, authorization, query limits, and audit events before external exposure.

### 6.5 High priority: monitoring and lineage

The current logger emits formatted text rather than structured JSON; telemetry is mainly local request timing and lexical overlap.

Adopt OpenTelemetry tracing plus Prometheus/Grafana metrics, and MLflow or an equivalent lightweight registry for experiments/artifacts.

Track:

- Filing discovery/availability-to-ready lag, extraction coverage, stale-cache age, source errors, and synthetic-data exclusions.
- Per-stage latency, retries, queue wait, 429s, actual fallback mode, endpoint/model version, and cost.
- Recall regressions, citation support, abstention, unsupported numbers, and corpus freshness.
- Sentiment coverage, class drift, calibration drift, backend changes, and feature distribution shifts.
- Delayed economic outcomes separately from online availability/quality metrics.

Do not log high-cardinality query strings or ticker/request IDs as unrestricted metric labels; keep identifiers in traces and structured logs. Version source files, parser/chunker, tokenizer/model, prompt, index, market snapshot, and backtest specification together.

SEC ingestion should use an identified user agent, caching, backoff, and a shared rate limiter across workers. SEC's current guidance caps automated access at 10 requests per second per user regardless of the number of machines; operate conservatively below that shared ceiling. [web:16][web:17]

## 7. Prioritized Delivery Roadmap

Effort ranges are planning estimates, not measured implementation commitments; adjust after a short implementation spike.

| Phase | Priority | Main deliverables | Acceptance gate | Indicative effort |
|---|---|---|---|---|
| 0: Research integrity | High | Explicit modes; no generated research labels; real/demo cache isolation; truthful stage status; remove unlabeled simulated history | No synthetic observation can enter research datasets; failures remain visible | 3–5 engineering days |
| 1: Reproducible foundation | High | Accession-level storage, availability timestamps, correct packaging, locked image, isolated tests | Clean install and rerun reproduce records/features; temporal/provenance tests pass | 1–2 weeks |
| 2: Retrieval quality | High/Medium | Structural parsing, stable chunks, hybrid search, reranker contracts, gold benchmark | Held-out relevance and citation metrics beat baseline without unacceptable latency/cost regression | 1–2 weeks |
| 3: Sentiment and factor research | High | Coverage-aware probabilities, labeled SEC evaluation, real historical panel, leakage-safe baselines | Reliable sentiment and separately reported out-of-sample economics; no profitability presumption | 2–4+ weeks plus data/labeling time |
| 4: Operational hardening | Medium | Background jobs, API boundary, traces, alerts, load tests, rollback | Defined latency/freshness/error objectives met under expected concurrency and fault injection | 1–2 weeks |
| 5: Optional scale | Low until justified | Dedicated vector service, self-hosted NIM/GPU tuning, Kubernetes/Helm if needed | Demonstrated scale or unit-cost bottleneck and tested recovery plan | Demand-driven |

### Release gates

- Integrity: no unlabeled synthetic data, no cross-issuer fallback, and complete source lineage.
- Reproducibility: versioned artifacts, stable IDs, clean package installation, and deterministic fixture tests.
- Retrieval: held-out passage relevance, numerical correctness, citation support, and abstention evaluation.
- Sentiment: SEC-domain labels, section coverage, calibrated outputs, and explicit backend identity.
- Economic research: point-in-time execution, complete forward labels, costs, exposure controls, and untouched holdout.
- Operations: verified runtime status, component health, latency/error budgets, monitoring, and reversible deployment/index cutovers.
