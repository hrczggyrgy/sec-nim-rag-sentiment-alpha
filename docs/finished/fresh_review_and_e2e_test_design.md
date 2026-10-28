# Fresh Review and End-to-End Bug-Finding Kit

## Scope and honesty

Repository: `hrczggyrgy/sec-nim-rag-sentiment-alpha`.
Pinned review target: `e4c79dc4dc4d069975c450c229fcc1dde5d7dd21` (October 28, 2026).
Previous target: `68fff3ca135f3edb9827d9ffeac026e8506eb32d`.

This kit is an unexecuted test design and candidate regression implementation against inspected APIs. Its Python syntax was checked, but the repository, its dependencies, Docker image, browser, and provider services were not run in the review environment. No passing/failing test count is claimed. No files were pushed to GitHub.

Tests intentionally assert desired integrity behavior, not today's buggy behavior; expect failures on the pinned commit. The production-key requirement, null sentiment, and missing partial-horizon labels are proposed contracts that should be accepted or revised explicitly before implementation. These tests are not marked xfail: silent acceptance of known regressions defeats the purpose. A changed contract can legitimately require changing a test, but never merely to make a build green.

No finite suite finds all bugs. The goal is to reproduce known defects, expose high-risk interactions, and prevent regression. Component tests, real local integration, UI tests, packaging tests, fault injection, and controlled live contracts complement rather than substitute for one another.

## What improved since the previous review

- Package moved to `src/sec_nim_alpha` and package discovery updated.
- Empty API keys can explicitly select offline behavior.
- Hash vectors now use SHA256 and are stable across processes (but remain nonsemantic).
- Research/production market-data exceptions no longer fabricate metrics.
- Random historical-panel generation removed.
- Sentiment now uses full class distributions and section-aware sampling.
- Per-query execution manifest and degraded generation status added.
- Container now uses multiple stages and a non-root user.
- Latest patch fixes Plotly's `secondary_y` argument.

These are useful changes, but several commit-message claims are broader than the implementation.

## Bug register

Evidence below identifies exact inspected files/functions at the pinned commit. C = defect directly supported by inspected control flow; V = environment-dependent risk requiring execution; P = proposed invariant beyond current behavior. “Expected test” is not an executed result.

| ID | Priority / evidence | Defect or risk | Reproducer and expected oracle |
|---|---|---|---|
| B01 | P0 / C | `app/main.py` imports removed `src.utils` inside the answer-button branch despite importing the replacement at the top. | Click Retrieve & Synthesize in a clean installation; no import exception. |
| B02 | P1 / C | `create_sentiment_volatility_figure()` compares `None >= 0`; historical rows legitimately have missing MOI. | Two historical filings, first MOI None; render chart without exception or invented score. NaN should also have neutral/missing styling, not bearish styling. |
| B03 | P1 / C | Market status enums are compared with literal strings in `app/main.py`. | SUCCESS must not show a market warning; DEGRADED must show the intended degraded state. |
| B04 | P2 / C | Truthiness hides valid 0.0 volatility/return as N/A. | Valid zero metrics display 0.00%, not missing. |
| B05 | P1 / C | Demo fallback still assigns NVDA CIK/accession/text to unsupported requested tickers. | Offline UNKNOWN request rejects unsupported issuer or displays true fixture identity. |
| B06 | P1 / C | Chunk IDs include only the first eight accession characters, which are commonly identical across an issuer's filings. | Same issuer, two years: chunk-ID sets must be disjoint. |
| B07 | P1 / C | `Chroma.from_documents()` receives no stable IDs; stable collection naming alone is not idempotent indexing. | Index identical documents twice: physical count unchanged, no duplicate citations. Confirm against installed Chroma adapter. |
| B08 | P1 / C | Collection identity uses configured NVIDIA model even when actual embedding backend is SHA256; parser/content/model revision also omitted. | Change actual backend/dimension/preprocessing: distinct index or explicit validated migration, never mixed embeddings. |
| B09 | P1 / C | PRODUCTION with absent/invalid key silently initializes hash/extractive fallback. | Proposed strict production policy: explicit initialization failure unless approved local semantic fallback is configured. |
| B10 | P1 / C | Execution manifest hardcodes sentiment and market statuses to SUCCESS; it lacks their real status inputs. | No sentiment input or missing market data must not become success. |
| B11 | P2 / C | `retrieval_count` is `rerank_count + len(top_docs)`; actual candidate count is discarded. | Four retrieved/final docs report four, not eight; initial count must not be smaller than final count. |
| B12 | P1 / C | Uppercase parser pattern for Item 1A becomes `ITEM A`, not `ITEM 1A`. | Normal uppercase SEC headings parse successfully and stop at real end boundaries. |
| B13 | P1 / C | Proportional section fallback persists in research/production and is indexed under real Item labels. | Missing sections remain failed/missing; do not relabel half a filing as Item 1A. |
| B14 | P1 / C | If only one edgartools section is missing, the both-short condition bypasses fallback and can mark the empty section SUCCESS. | Missing MD&A with long risk text: MD&A extraction is not success; valid risk text retained. |
| B15 | P1 / C | Risk-only sampling reserves at least one MD&A slot and can relabel a risk sentence as MD&A. | >limit risk-only sentences: exact target sample count with zero MD&A coverage. |
| B16 | P2 / C | Sampling warning computes both counts after truncation: e.g. “2 of 2” instead of “2 of 4.” | Retain original eligible sentence count. |
| B17 | P1 / C | A malformed later FinBERT prediction leaves earlier scored items appended, then adds a full lexicon fallback batch. | Two inputs, valid first result then invalid second: exactly two final results, one backend, visible warning. |
| B18 | P1 / C | `zip()` silently accepts missing prediction rows; an empty list yields measured-looking neutral output with no failure warning. | Wrong response length or invalid distributions must trigger complete fallback/failure, not dropped observations. |
| B19 | P1 / C/P | No eligible sentences produces MOI 0.0 and neutral ratio 1.0. | Proposed missing-data contract: no observation is not measured neutrality; null score and unavailable status. |
| B20 | P1 / C | Market labels still use filing date, not actual availability/executable time; `filed_at` is date-only and acceptance is copied into public dissemination. | After-hours filing, holiday/weekend: execution obeys point-in-time rules; acceptance and public availability remain distinct. |
| B21 | P1 / C/P | Four-to-six returns publish numbers in seven-day fields as DEGRADED. | Proposed research label contract: no seven-day label until seven valid returns; partial diagnostics stored separately. |
| B22 | P1 / C | NaN/inf/nonpositive/duplicate/unsorted price observations are not validated; numerical failure can be marked SUCCESS. | Invalid bars fail or normalize under an explicit rule; no nonfinite successful metrics. |
| B23 | P1 / C | Numeric “claim support” is substring presence: 5% matches 25%; sign, issuer, metric and period ignored. | Contradictory metric/sign/entity cases never count as supported claims. |
| B24 | P1 / C | Empty retrieval still calls the LLM; irrelevant nonempty retrieval never triggers insufficiency. | No-evidence generator call count zero; unanswerable benchmark requires abstention. |
| B25 | P2 / C | JSON logging interpolates unescaped quotes/newlines/backslashes. | Every emitted log record parses as one JSON object, including provider error strings. |
| B26 | P1 / C | `from_dict()` mutates caller data; demo cache reads/JSON errors occur outside recovery try-block; latest cache uses mtime. | Repeat deserialization leaves payload unchanged; corrupt cache recovers; touching an old filing cannot make it newest. |
| B27 | P1 / C | UI updates last panel row regardless of accession; empty panel receives only metric columns, leaving ticker/date/accession missing. | Current filing inserted/updated by accession; old row identity never receives new or synthetic metrics. |
| B28 | P1 / C | Config creates `src/data` in source checkout or data beside installed package, not intended `/app/data`, during import. | Clean wheel import under non-root/read-only site-packages succeeds using an explicit writable DATA_DIR. |
| B29 | P1 / V | Builder installs editable package before copying `src/` or README; editable import mapping can be empty or build metadata may fail. | Clean image build followed by import from unrelated working directory. Exact failure depends on setuptools. |
| B30 | P1 / V | Healthcheck calls curl but runtime stage installs no curl. | Built image health becomes healthy; inspect command exit status and utility availability. |
| B31 | P1 / C | `.dockerignore` excludes uv.lock; Docker uses unlocked pip install and includes dev dependencies. | Frozen dependency build uses committed lock; runtime excludes development tools. |
| B32 | P1 / C | “Raw” artifact stores only extracted/truncated sections; hash is computed before per-section truncation. | Raw file byte-identical to source; stored parsed content/hash consistent; source offsets actually resolve. |
| B33 | P1 / C | Cached `run_full_pipeline` wraps refresh flag; repeat refresh=True can reuse cache instead of refreshing; request ID also cached. | Each refresh event performs required fresh I/O; each query gets new request ID, unchanged queries do not re-embed. |
| B34 | P2 / C/V | Timeout/retry/rate/TTL configuration is not wired explicitly in inspected call sites; dependency defaults may act separately. | Force transient/terminal failures and measure total attempts/deadline; inspect shared SEC request budget. |

Severity: P0 blocks a central user journey; P1 blocks correctness/research release; P2 affects presentation, observability, or operational reliability. An apparently cosmetic error becomes P1 when it conceals missing or invalid research data.

## Included tests versus remaining design

Included runnable candidates cover removed imports, enum misuse, missing chart values, accession collisions, production-policy failure, issuer substitution, missing/nonfinite/partial market data, synthetic record rejection, serialization, corrupt/stale caches, uppercase/missing sections, sampling, malformed FinBERT batches, manifests/counts, empty evidence, false numeric support, JSON logging, actual local Chroma indexing, real market arithmetic with frozen bars, and Streamlit interactions.

The local happy-path integration replaces only EDGAR and Yahoo external boundaries, then runs actual loader serialization, chunking, lexicon sentiment, Chroma embeddings/search, extractive synthesis, and response serialization. Hash retrieval proves plumbing, not semantic accuracy. A separate semantic benchmark below is required.

The kit does NOT yet implement every matrix scenario, browser test, concurrent writer test, live provider contract, or new alpha pipeline. This is a staged implementation plan plus concrete regression tests, not a claim of complete coverage.

## End-to-end architecture and oracles

```text
Clean build/install -> config + writable storage -> fake/real SEC transport
  -> actual filing loader -> parser + provenance -> immutable cache
  -> actual feature/chunk pipeline -> actual local vector DB
  -> rerank/generator boundary -> response + execution manifest
  -> Streamlit app interactions -> browser/container lifecycle
```

Use independent oracles:

- Fixture corpus has explicit expected issuer, accession, sections, offsets, hashes and required evidence IDs.
- Eight frozen closes `[100,101,102,100,103,104,105,106]` imply 6.00% return; volatility formula fixed to sample standard deviation of seven simple returns times sqrt(252).
- Zero is valid, missing is null, simulated is explicitly excluded from research, partial is not complete.
- Gold answers/claims authored independently of generated outputs; presence of a citation is not proof of support.
- Each stage records actual backend, artifact version, duration and failure state. Do not infer successful execution merely from constructing a client.

## Scenario matrix to implement beyond the included regressions

### A. Install/container gate

1. Build wheel and sdist; install wheel in a clean venv, change outside checkout, import package, run smoke test.
2. Install/run as non-root with read-only site-packages and explicit writable data path.
3. Run supplied container gate; test image health, termination, restart and mounted-storage persistence.
4. Check image for credentials, forbidden build artifacts, unused development tools; generate dependency/SBOM report.
5. Repeat build with lock enforcement; mismatched lock must fail loudly rather than resolve silently.

### B. SEC parsing and cache

- Normal/mixed/uppercase headings, table of contents duplicates, headings across HTML elements, Unicode whitespace, missing Item 7, missing Item 1A, long >100k-character filings, tables and footnotes.
- Roundtrip source bytes, offsets, full content hash; parser-version change invalidates derived artifacts only.
- New accession, old accession touched, expired TTL, corrupt/truncated JSON, incompatible schema, interrupted write, concurrent same-accession writes.
- Demo/live namespaces; filename alone cannot certify a record as genuine. Reject issuer/CIK/accession mismatch inside cached payload.
- Invalid input/path traversal ticker (such as '../outside', glob characters, empty or too-long values); no file access outside configured storage.
- Cache-read permission errors must be distinguishable from SEC outages; cache-write errors must not silently replace a successfully fetched live filing with synthetic content.

### C. Market data and time

- 0, 1, 4, 6, 7 and 8 valid closes; wrong/missing Close column; null, NaN, inf, zero, negative prices; duplicates and descending timestamps.
- Premarket, intraday, after-hours, Friday, exchange holiday and DST transition.
- Late ingestion: execution cannot predate signal readiness. Distinguish acceptance, dissemination and observation timestamps.
- Corporate actions, point-in-time ticker mapping, delisted securities and incomplete future horizon.
- Freeze clock and forbid reads beyond requested as-of for feature creation. Future labels belong to offline evaluation, never prompts/features.

### D. Sentiment

- Both sections, one section only, empty sections; 199/200/201 eligible sentences; heavily imbalanced section counts.
- Prediction row count/order mismatch, unknown labels, duplicated labels, missing class, invalid probability sums, negative/nonfinite values, first/middle/last batch failure.
- Exactly one score per sampled sentence; section coverage and aggregate denominator equal sample size; no mixed-backend partial results.
- Negation, increasing losses versus increasing revenue, conditionals, dates, decimals, U.S./Inc. sentence boundaries, repeated boilerplate, 512-token overflow.
- In production, deliberate lexicon selection and failed FinBERT fallback must be distinguishable policies, not silently equivalent.

### E. Retrieval/NIM

- Stable IDs, identical repeated ingestion, partial reindex crash, resume, deleted/changed chunk, restart, concurrent updates.
- Same accession with different embedding backend/dimension or parser/chunker revision: no shared incompatible collection.
- Semantic quality uses actual pinned semantic embeddings, not hash vectors. Gold passage recall, irrelevant-query abstention, citation issuer/date correctness.
- Reranker: valid reorder, empty output, duplicate docs, unknown document IDs, malformed scores, 401/403/404/429/5xx and timeout.
- Generator: valid answer, empty content, invented citation, unsupported number, refusal, prompt injection, timeout, stream abort.
- Request-local status after success->failure->success; parallel requests cannot overwrite one another's manifest.
- No evidence means no LLM call; cross-issuer/as-of filters cannot be bypassed by query wording.
- Count calls, retries, token/cost budget and deadlines; no paid live calls on untrusted PRs.

### F. UI journey

Use Streamlit AppTest for script execution/widget state, then Playwright against the built app for actual browser/network/DOM integration. AppTest explicitly runs the app without a browser; do not describe it as a complete browser E2E test. [web:30][web:31]

1. Fresh DEMO load with no key and explicit provider failure: labeled demo, no exception.
2. Select RESEARCH; complete controlled fixture flows to correct issuer metrics and cited answer.
3. Select PRODUCTION; missing required services yields a visible controlled error, not raw traceback or silent hash fallback.
4. Click answer and verify full manifest, current accession, answer and citation passages.
5. Switch issuer/year; no previous issuer citations or sentiment survive.
6. Force refresh twice, rerun after checkbox toggles, warm query and app restart; inspect provider/embed call counters.
7. Zero and null metrics; success/degraded/failed warnings; missing historical MOI; empty historical panel.
8. Two browser contexts with different credentials; resource cache cannot cross user/tenant policy boundaries.
9. Query containing markup/injection; no scripts execute, filing text cannot override privileged instructions.
10. Capture screenshot, browser console, server exception and Playwright trace on failure. Clear resource/data caches between isolation tests; preserve them only in tests explicitly targeting cache behavior.

### G. Research/alpha gate

The current repository has no complete historical factor/portfolio backtester to exercise end to end. Do not invent one and report it passing.

When implemented, use a small golden panel with known event times and expected signal/label arrays. Assert no future rows accessible to transforms; chronological fit/validation/test; purge overlapping labels; point-in-time membership; transaction costs; benchmark exposures; deterministic portfolio arithmetic. Add a deliberately leaky feature and confirm a leakage sentinel fails. Treat profitability as a research outcome, not a guaranteed software test.

## Fault matrix and property tests

Use targeted cases plus pairwise combinations across execution mode, issuer, cache state, section completeness, price horizon, FinBERT outcome, embedding backend, reranker outcome and generator outcome. Add mandatory triple-fault cases:

- Fresh research filing + market failure + successful LLM: valid answer, failed market status, no label fabricated.
- Partial FinBERT batch + reranker timeout + LLM timeout: single-backend sentiment, degraded reranker/generator, correct warnings/counts.
- Corrupt stale cache + unavailable EDGAR + production mode: controlled failure, no demo substitution.
- Parallel same-accession indexing + process restart + concurrent query: no duplicates, corruption or mixed snapshot.

Use Hypothesis for legal ticker/path handling, serialization purity, finite normalized distributions, unique IDs, monotonic event-time constraints and missing-value propagation. Use mutation testing after baseline stabilization: remove a synthetic-data guard, swap section labels, duplicate a chunk, change status to SUCCESS, shift event time backward. The suite must detect those specific seeded faults. A coverage percentage alone does not establish bug-finding power.

## How to run the supplied candidates

Extract this archive into a checkout so `qa/` is at the repository root. Use a disposable checkout at the pinned commit; these tests write only temporary test data after imports, but current config has import-time directory side effects.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]' pytest-socket
mkdir -p qa-artifacts
python -m pytest -c qa/pytest.ini qa/tests/test_regressions.py \
  --disable-socket --junitxml=qa-artifacts/regressions.xml
python -m pytest -c qa/pytest.ini qa/tests/test_local_e2e.py qa/tests/test_ui.py \
  --disable-socket --junitxml=qa-artifacts/local-e2e.xml
bash qa/check_container.sh
```

These setup commands resolve dependencies and are for initial reproduction, not a claim of locked reproducibility. Add pytest-socket to dev dependencies and update/validate the lock before the CI frozen-install gate.

pytest-socket blocks Python socket calls, but use process/container-level egress controls as the definitive offline boundary; imports, teardown, subprocesses and native libraries need separate attention. Do not allow all networking just to suppress a telemetry failure. [web:34]

For native Chroma/DNS restrictions, use an egress-denied sandbox that still permits required local IPC. Browser E2E needs loopback/local bridge traffic, not public internet. Build/download dependencies in a separate permitted setup stage; test stage has no credentials or external egress.

## CI rollout

### Every PR

- Clean locked setup, syntax/lint/type checks.
- Network-denied component/local integration/AppTest suites.
- Wheel installation outside checkout; clean Docker build/import/health gate.
- Gold retrieval/sentiment evaluation with fixed local artifacts.
- Publish JUnit, traces, sanitized manifests, logs and coverage even after failures.

### Nightly or manual

- Browser E2E, restart/concurrency/fault-injection cases and mutation tests.
- Protected, opt-in external contract probes with per-run call/token budget, redacted secrets and bounded issuer list.
- Validate actual model capability/latency and failure policy. External model variability means assert schema/provenance/evidence quality, not identical prose.

### Release blockers

- Any critical path exception, unlabeled synthetic research record, issuer/accession mismatch, future-data leakage, duplicate persistent indexing, false-success manifest, nonfinite successful label, or failed clean image startup.
- Unexpected new skip/xfail and unregistered markers fail review.
- No assertion of “all bugs fixed”; publish tested commit/environment, measured outcomes and known untested boundaries.

## Recommended fix order

1. B01: answer-button import; B02/B03/B04: chart/enum/zero UI regressions.
2. B05/B06/B07/B08: issuer identity, full accession chunk IDs, actual-backend index identity and idempotent upserts.
3. B09/B10/B20/B21/B22: strict policy, honest status, event time and valid complete labels.
4. B15–B19: transactional FinBERT results, schema validation, sampling correctness and null sentiment.
5. B26–B33: atomic cache/data lineage, container build/runtime, refresh/request tracing.
6. Add semantic evaluation, complete fault matrix and eventually true point-in-time alpha testing.
