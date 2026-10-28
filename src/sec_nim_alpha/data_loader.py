"""Automated SEC EDGAR filing ingestion and Yahoo Finance market data alignment."""

import json
import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd
import yfinance as yf
from sec_nim_alpha.config import CACHE_DIR, RAW_FILINGS_DIR, CONFIG, ensure_data_dirs
from sec_nim_alpha.schemas import (
    FilingIdentifier,
    MarketMetrics,
    SectionExtraction,
    ExecutionMode,
    ComponentStatus,
    compute_content_hash,
)
from sec_nim_alpha.utils import get_logger

logger = get_logger(__name__)


# Curated demo corpus - clearly labeled as synthetic
DEMO_SEC_CORPUS: dict[str, dict[str, str]] = {
    "NVDA": {
        "filing_date": "2026-02-26",
        "accession": "0001045810-26-000014",
        "cik": "0001045810",
        "risk_factors": (
            "Item 1A. Risk Factors. Our Data Center revenue growth depends heavily on cloud service "
            "providers and hyperscale customers deploying Blackwell and Rubin GPU architectures. "
            "Supply chain concentration at TSMC for advanced CoWoS packaging and HBM3e/HBM4 memory "
            "creates lead-time vulnerabilities. Export controls enforced by the U.S. Department of "
            "Commerce restrict shipments of high-density accelerators to certain international markets, "
            "which may adversely impact regional gross margins and long-term demand visibility."
        ),
        "mda": (
            "Item 7. Management's Discussion and Analysis of Financial Condition. Full-year Data Center "
            "revenue increased 114% year-over-year driven by enterprise adoption of NVIDIA NIM inference "
            "microservices and sovereign AI infrastructure buildouts. Gross margins expanded to 75.2% "
            "due to strong software attach rates via NVIDIA AI Enterprise. While custom ASIC competition "
            "is increasing among hyperscalers, our CUDA software ecosystem and NVLink fabric maintain "
            "significant total cost of ownership advantages across training and inference workloads."
        ),
    },
    "AAPL": {
        "filing_date": "2025-11-03",
        "accession": "0000320193-25-000106",
        "cik": "0000320193",
        "risk_factors": (
            "Item 1A. Risk Factors. Global macroeconomic headwinds, foreign exchange fluctuations, and "
            "regulatory scrutiny under the EU Digital Markets Act could materially impact App Store "
            "commission structures. Hardware manufacturing remains concentrated in East Asia, exposing "
            "quarterly hardware shipments to tariff adjustments and component shortages."
        ),
        "mda": (
            "Item 7. Management's Discussion and Analysis. Services revenue achieved an all-time record "
            "with over 1 billion paid subscriptions across our installed base of active devices. "
            "On-device Apple Intelligence features drove stronger-than-expected upgrade cycles in "
            "premium iPhone Pro tiers, expanding consolidated operating margin by 140 basis points."
        ),
    },
    "MSFT": {
        "filing_date": "2025-07-30",
        "accession": "0000950170-25-098211",
        "cik": "0000950170",
        "risk_factors": (
            "Item 1A. Risk Factors. Rapid expansion of global AI data center capacity requires substantial "
            "capital expenditures in power grid commitments and GPU clusters, which may compress near-term "
            "free cash flow conversion if enterprise Copilot seat adoption decelerates."
        ),
        "mda": (
            "Item 7. Management's Discussion and Analysis. Microsoft Cloud revenue surpassed $140 billion "
            "annualized run rate, with Azure growth accelerated by 11 percentage points from AI services. "
            "Commercial bookings increased 24% on strong multi-year E5 and GitHub Copilot renewals."
        ),
    },
    "TSLA": {
        "filing_date": "2025-01-29",
        "accession": "0001628280-25-000012",
        "cik": "0001628280",
        "risk_factors": (
            "Item 1A. Risk Factors. Automotive revenue growth is dependent on continued demand for Model Y "
            "and Model 3 refresh, while Cybertruck ramp faces manufacturing complexity at the Texas Gigafactory. "
            "FSD regulatory approval timelines remain uncertain across jurisdictions. Battery raw material "
            "costs for lithium and nickel expose gross margins to commodity volatility."
        ),
        "mda": (
            "Item 7. Management's Discussion and Analysis. Total automotive deliveries exceeded 1.8 million "
            "vehicles, with energy storage deployments growing 125% year-over-year to 14.7 GWh. "
            "Operating margin improved to 8.2% on better fixed cost absorption and FSD revenue recognition. "
            "AI training cluster Cortex expansion positions us for accelerated autonomy development."
        ),
    },
    "AMZN": {
        "filing_date": "2025-02-06",
        "accession": "0001018724-25-000004",
        "cik": "0001018724",
        "risk_factors": (
            "Item 1A. Risk Factors. AWS growth rate deceleration risk from enterprise optimization cycles "
            "and competitive pressure from Microsoft Azure AI and Google Cloud Vertex AI. "
            "Fulfillment network capital intensity remains elevated with Project Kuiper satellite "
            "constellation deployment requiring multi-billion dollar annual investments."
        ),
        "mda": (
            "Item 7. Management's Discussion and Analysis. AWS revenue accelerated to 19% year-over-year "
            "growth driven by generative AI workloads on Bedrock and Trainium2 custom silicon adoption. "
            "North America retail operating income expanded 45% on regionalization efficiency gains. "
            "Advertising services surpassed $50 billion annualized run rate."
        ),
    },
}


@dataclass
class FilingRecord:
    """Structured container for an ingested SEC filing with full provenance."""

    # Filing identity
    ticker: str
    cik: str
    form_type: str
    filing_date: str
    accession_number: str
    period_end: str
    filed_at: str
    public_disseminated_at: Optional[str] = None
    ingestion_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    source_url: Optional[str] = None
    content_hash: Optional[str] = None

    # Extracted content
    risk_factors_text: str = ""
    mda_text: str = ""

    # Section extraction provenance
    risk_factors_extraction: Optional[SectionExtraction] = None
    mda_extraction: Optional[SectionExtraction] = None

    # Market metrics with provenance
    market_metrics: Optional[MarketMetrics] = None

    # Execution metadata
    is_synthetic_fallback: bool = False
    execution_mode: ExecutionMode = ExecutionMode.DEMO

    def to_dict(self) -> dict:
        """Serialize to dict for caching."""
        data = asdict(self)
        # Handle nested dataclasses
        if self.risk_factors_extraction:
            data["risk_factors_extraction"] = {
                "section_name": self.risk_factors_extraction.section_name,
                "content": self.risk_factors_extraction.content,
                "char_start": self.risk_factors_extraction.char_start,
                "char_end": self.risk_factors_extraction.char_end,
                "extraction_status": self.risk_factors_extraction.extraction_status.value,
                "parser_version": self.risk_factors_extraction.parser_version,
                "warnings": list(self.risk_factors_extraction.warnings),
            }
        if self.mda_extraction:
            data["mda_extraction"] = {
                "section_name": self.mda_extraction.section_name,
                "content": self.mda_extraction.content,
                "char_start": self.mda_extraction.char_start,
                "char_end": self.mda_extraction.char_end,
                "extraction_status": self.mda_extraction.extraction_status.value,
                "parser_version": self.mda_extraction.parser_version,
                "warnings": list(self.mda_extraction.warnings),
            }
        if self.market_metrics:
            data["market_metrics"] = self.market_metrics.to_dict()
        data["execution_mode"] = self.execution_mode.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "FilingRecord":
        """Deserialize from cached dict without mutating input."""
        import copy
        data = copy.deepcopy(data)
        # Handle nested objects
        if "risk_factors_extraction" in data and data["risk_factors_extraction"]:
            ext = data["risk_factors_extraction"]
            data["risk_factors_extraction"] = SectionExtraction(
                section_name=ext["section_name"],
                content=ext["content"],
                char_start=ext["char_start"],
                char_end=ext["char_end"],
                extraction_status=ComponentStatus(ext["extraction_status"]),
                parser_version=ext["parser_version"],
                warnings=tuple(ext.get("warnings", [])),
            )
        if "mda_extraction" in data and data["mda_extraction"]:
            ext = data["mda_extraction"]
            data["mda_extraction"] = SectionExtraction(
                section_name=ext["section_name"],
                content=ext["content"],
                char_start=ext["char_start"],
                char_end=ext["char_end"],
                extraction_status=ComponentStatus(ext["extraction_status"]),
                parser_version=ext["parser_version"],
                warnings=tuple(ext.get("warnings", [])),
            )
        if "market_metrics" in data and data["market_metrics"]:
            mm = data["market_metrics"]
            data["market_metrics"] = MarketMetrics(
                ticker=mm["ticker"],
                filing_date=mm["filing_date"],
                post_7d_return_pct=mm.get("post_7d_return_pct"),
                post_7d_volatility_pct=mm.get("post_7d_volatility_pct"),
                trading_days_observed=mm.get("trading_days_observed", 0),
                required_trading_days=mm.get("required_trading_days", 7),
                status=ComponentStatus(mm.get("status", "success")),
                error_message=mm.get("error_message"),
                data_source=mm.get("data_source", "yfinance"),
                fetched_at=mm.get("fetched_at", datetime.utcnow().isoformat() + "Z"),
            )
        data["execution_mode"] = ExecutionMode(data.get("execution_mode", "demo"))
        return cls(**data)


class SECDataLoader:
    """Ingests SEC 10-K filings via edgartools and aligns post-filing equity volatility."""

    def __init__(self, user_agent: Optional[str] = None, execution_mode: Optional[ExecutionMode] = None) -> None:
        self.user_agent = user_agent or CONFIG.sec_user_agent
        self.execution_mode = execution_mode or CONFIG.execution_mode
        ensure_data_dirs()

    def fetch_market_metrics(self, ticker: str, filing_date: str) -> MarketMetrics:
        """
        Calculates 7-trading-day cumulative return and annualized volatility after filing date.
        Returns MarketMetrics with explicit status and completeness flags.
        """
        try:
            start_dt = datetime.strptime(filing_date, "%Y-%m-%d")
            end_dt = start_dt + timedelta(days=25)  # Buffer for weekends/holidays
            hist = yf.Ticker(ticker).history(
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
            )
            if hist.empty:
                raise ValueError("No market price bars returned from Yahoo Finance.")

            # Validate price data
            closes = hist["Close"]
            if closes.isna().any():
                raise ValueError("Price data contains NaN values")
            if np.isinf(closes).any():
                raise ValueError("Price data contains infinite values")
            if (closes <= 0).any():
                raise ValueError("Price data contains non-positive values")
            # Check for duplicates in index
            if closes.index.duplicated().any():
                raise ValueError("Price data contains duplicate timestamps")
            # Check for unsorted index
            if not closes.index.is_monotonic_increasing:
                raise ValueError("Price data timestamps are not sorted")

            # Use first 8 trading days to get 7 daily returns
            closes = closes.iloc[:8]
            daily_returns = closes.pct_change().dropna()
            trading_days_observed = len(daily_returns)

            # In research/production mode, require complete 7-day horizon
            if self.execution_mode in (ExecutionMode.RESEARCH, ExecutionMode.PRODUCTION):
                if trading_days_observed < 7:
                    raise ValueError(f"Incomplete horizon: {trading_days_observed} trading days observed (need 7)")

            cum_return_pct = float(((closes.iloc[-1] / closes.iloc[0]) - 1.0) * 100.0)
            ann_volatility_pct = float(daily_returns.std() * np.sqrt(252) * 100.0)

            status = ComponentStatus.SUCCESS
            error_message = None
            
            # In demo mode, allow degraded for partial horizon
            if self.execution_mode == ExecutionMode.DEMO and trading_days_observed < 7:
                status = ComponentStatus.DEGRADED
                error_message = f"Only {trading_days_observed} trading days observed; horizon incomplete"

            return MarketMetrics(
                ticker=ticker,
                filing_date=filing_date,
                post_7d_return_pct=round(cum_return_pct, 2),
                post_7d_volatility_pct=round(ann_volatility_pct, 2),
                trading_days_observed=trading_days_observed,
                status=status,
                error_message=error_message,
            )

        except Exception as exc:
            logger.warning("Market data fetch failed for %s (%s): %s", ticker, filing_date, exc)
            if self.execution_mode == ExecutionMode.RESEARCH or self.execution_mode == ExecutionMode.PRODUCTION:
                # In research/production mode, return incomplete metrics with explicit failure status
                return MarketMetrics(
                    ticker=ticker,
                    filing_date=filing_date,
                    post_7d_return_pct=None,
                    post_7d_volatility_pct=None,
                    trading_days_observed=0,
                    status=ComponentStatus.FAILED,
                    error_message=str(exc),
                )
            else:
                # Demo mode: seeded deterministic fallback with explicit warning
                seed = sum(ord(c) for c in f"{ticker}{filing_date}")
                rng = np.random.default_rng(seed)
                return MarketMetrics(
                    ticker=ticker,
                    filing_date=filing_date,
                    post_7d_return_pct=round(float(rng.uniform(-4.5, 8.5)), 2),
                    post_7d_volatility_pct=round(float(rng.uniform(22.0, 48.0)), 2),
                    trading_days_observed=0,
                    status=ComponentStatus.OFFLINE_FALLBACK,
                    error_message=f"Demo fallback: {exc}",
                )

    def _extract_sections(self, full_text: str) -> tuple[SectionExtraction, SectionExtraction]:
        """Extract Item 1A and Item 7 sections from filing text with status tracking."""
        risk_factors_ext = self._extract_section(full_text, "Item 1A", "Risk Factors")
        mda_ext = self._extract_section(full_text, "Item 7", "Management's Discussion")
        return risk_factors_ext, mda_ext

    def _extract_section(self, text: str, item_marker: str, section_title: str) -> SectionExtraction:
        """Extract a specific section with boundary detection."""
        # Try to find section boundaries - handle both normal and uppercase SEC headings
        # Item 1A -> ITEM 1A (not ITEM A)
        item_num = item_marker.replace("Item ", "").replace("item ", "")
        start_markers = [
            f"{item_marker}. {section_title}",
            f"{item_marker} {section_title}",
            f"ITEM {item_num}. {section_title.upper()}",
            f"ITEM {item_num} {section_title.upper()}",
        ]
        # Include both normal and uppercase end markers
        end_markers = [
            "Item 1B", "ITEM 1B",
            "Item 2.", "ITEM 2.",
            "Item 3.", "ITEM 3.",
            "Item 4.", "ITEM 4.",
            "Item 5.", "ITEM 5.",
            "Item 6.", "ITEM 6.",
            "Item 7.", "ITEM 7.",
            "Item 7A.", "ITEM 7A.",
            "Item 8.", "ITEM 8.",
        ]

        content = ""
        char_start = -1
        char_end = -1
        status = ComponentStatus.FAILED
        warnings = []

        for marker in start_markers:
            idx = text.find(marker)
            if idx >= 0:
                char_start = idx
                # Find end
                remaining = text[idx:]
                end_idx = len(remaining)
                for end_marker in end_markers:
                    if end_marker == f"{item_marker}.":
                        continue
                    found = remaining.find(end_marker)
                    if found > 100 and found < end_idx:  # Must be after some content
                        end_idx = found
                content = remaining[:end_idx].strip()
                char_end = char_start + len(content)
                status = ComponentStatus.SUCCESS
                break

        if status == ComponentStatus.FAILED:
            if self.execution_mode in (ExecutionMode.RESEARCH, ExecutionMode.PRODUCTION):
                # In research/production, missing sections are failures - no proportional fallback
                warnings.append(f"Section boundary detection failed for {item_marker}; no fallback in {self.execution_mode.value} mode")
                content = ""
                char_start = -1
                char_end = -1
                status = ComponentStatus.FAILED
            else:
                # Demo mode: proportional split with warning
                warnings.append(f"Section boundary detection failed for {item_marker}; using proportional fallback")
                mid = len(text) // 2
                if item_marker == "Item 1A":
                    content = text[:mid].strip()
                    char_start = 0
                    char_end = mid
                else:
                    content = text[mid:].strip()
                    char_start = mid
                    char_end = len(text)
                status = ComponentStatus.DEGRADED

        return SectionExtraction(
            section_name=f"{item_marker}: {section_title}",
            content=content,
            char_start=char_start,
            char_end=char_end,
            extraction_status=status,
            parser_version="1.0.0",
            warnings=tuple(warnings),
        )

    def _fetch_live_10k(self, ticker: str) -> FilingRecord:
        """Fetch live 10-K from EDGAR with full provenance."""
        from edgar import Company, set_identity

        set_identity(self.user_agent)
        company = Company(ticker)
        filings = company.get_filings(form="10-K")
        if not filings or len(filings) == 0:
            raise RuntimeError(f"No 10-K filings found on EDGAR for {ticker}")

        latest = filings[0]
        filing_date = str(latest.filing_date)
        accession = str(latest.accession_no)
        cik = str(latest.cik)
        period_end = str(getattr(latest, "period_end", filing_date))
        filed_at = str(getattr(latest, "filing_date", filing_date))
        accepted_at = str(getattr(latest, "accepted_date", None)) if hasattr(latest, "accepted_date") else None

        tenk_obj = latest.obj()
        risk_text = str(getattr(tenk_obj, "risk_factors", "") or "")
        mda_text = str(getattr(tenk_obj, "mda", "") or "")

        # If either section is too short, try full text extraction for both
        # Use full text without arbitrary character limit
        if len(risk_text) < 200 or len(mda_text) < 200:
            full_text = latest.text()  # No cap - store full original
            risk_ext, mda_ext = self._extract_sections(full_text)
            risk_text = risk_ext.content
            mda_text = mda_ext.content
        else:
            risk_ext = SectionExtraction(
                section_name="Item 1A: Risk Factors",
                content=risk_text,
                char_start=0,
                char_end=len(risk_text),
                extraction_status=ComponentStatus.SUCCESS,
                parser_version="1.0.0",
            )
            mda_ext = SectionExtraction(
                section_name="Item 7: MD&A",
                content=mda_text,
                char_start=0,
                char_end=len(mda_text),
                extraction_status=ComponentStatus.SUCCESS,
                parser_version="1.0.0",
            )

        # Compute content hash for integrity (use same format as raw storage: double newline)
        combined_content = f"{risk_text}\n\n{mda_text}"
        content_hash = compute_content_hash(combined_content)

        market_metrics = self.fetch_market_metrics(ticker, filing_date)

        # Store truncated text in record (for memory), but keep full text for raw file
        risk_text_stored = risk_text[:50000]
        mda_text_stored = mda_text[:50000]

        record = FilingRecord(
            ticker=ticker,
            cik=cik,
            form_type="10-K",
            filing_date=filing_date,
            accession_number=accession,
            period_end=period_end,
            filed_at=filed_at,
            public_disseminated_at=accepted_at,
            source_url=f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}",
            content_hash=content_hash,
            risk_factors_text=risk_text_stored,
            mda_text=mda_text_stored,
            risk_factors_extraction=risk_ext,
            mda_extraction=mda_ext,
            market_metrics=market_metrics,
            is_synthetic_fallback=False,
            execution_mode=self.execution_mode,
        )
        
        # Save raw filing with full untruncated content for integrity verification
        # Use double newline separator to match content hash computation
        raw_file = RAW_FILINGS_DIR / f"{record.ticker}_{record.accession_number}_raw.txt"
        raw_content = f"{risk_text}\n\n{mda_text}"
        raw_file.write_text(raw_content, encoding="utf-8")
        
        return record

    def fetch_latest_10k(self, ticker: str, force_refresh: bool = False) -> FilingRecord:
        """
        Fetches the latest 10-K filing from SEC EDGAR or falls back to curated demo data.
        Cache is keyed by accession for immutable isolation.
        """
        ticker = ticker.upper().strip()

        # Try live fetch first (unless demo mode and cache exists)
        if self.execution_mode == ExecutionMode.DEMO and not force_refresh:
            # Check for cached live filing by accession
            cached_files = list(CACHE_DIR.glob(f"{ticker}_*_10k.json"))
            if cached_files:
                # Load most recent cached filing by accession number (not mtime)
                # Parse accession from filename: {ticker}_{accession}_10k.json
                def extract_accession(f):
                    name = f.name
                    prefix = f"{ticker}_"
                    suffix = "_10k.json"
                    if name.startswith(prefix) and name.endswith(suffix):
                        return name[len(prefix):-len(suffix)]
                    return ""
                
                # Filter out demo cache files (they have "DEMO" in accession position)
                live_cache_files = [f for f in cached_files if extract_accession(f) and "DEMO" not in extract_accession(f)]
                
                if live_cache_files:
                    # Sort by accession (which contains date info) descending
                    latest_cache = max(live_cache_files, key=lambda f: extract_accession(f))
                    
                    # Try to load cache with corrupt cache recovery
                    try:
                        logger.info("Loading cached 10-K filing for %s from %s", ticker, latest_cache)
                        data = json.loads(latest_cache.read_text(encoding="utf-8"))
                        return FilingRecord.from_dict(data)
                    except (json.JSONDecodeError, ValueError, KeyError) as exc:
                        logger.warning("Corrupt cache file %s (%s); attempting live fetch", latest_cache, exc)
                        # Fall through to live fetch

        try:
            record = self._fetch_live_10k(ticker)

            # Reject synthetic records in research/production mode
            if self.execution_mode in (ExecutionMode.RESEARCH, ExecutionMode.PRODUCTION) and record.is_synthetic_fallback:
                raise ValueError("Research/production mode received synthetic fallback from live fetch")

            # Cache by accession (immutable)
            cache_file = CACHE_DIR / f"{record.ticker}_{record.accession_number}_10k.json"
            cache_file.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")

            # Raw filing is already saved in _fetch_live_10k with full content

            logger.info("Successfully ingested live SEC 10-K for %s (%s, accession: %s)",
                       ticker, record.filing_date, record.accession_number)
            return record

        except Exception as exc:
            logger.warning("Live EDGAR fetch unavailable for %s (%s).", ticker, exc)

            if self.execution_mode == ExecutionMode.RESEARCH or self.execution_mode == ExecutionMode.PRODUCTION:
                # In research/production mode, fail explicitly
                raise RuntimeError(f"Live EDGAR fetch required but failed: {exc}") from exc

            # Demo mode: use curated corpus with explicit labeling
            if ticker not in DEMO_SEC_CORPUS:
                # Reject unsupported tickers in demo mode rather than silently substituting
                raise ValueError(
                    f"Ticker '{ticker}' not in demo corpus. "
                    f"Supported demo tickers: {', '.join(sorted(DEMO_SEC_CORPUS.keys()))}. "
                    f"Use a supported ticker or enable live EDGAR fetch."
                )

            fallback = DEMO_SEC_CORPUS[ticker]

            filing_date = fallback["filing_date"]
            market_metrics = self.fetch_market_metrics(ticker, filing_date)

            combined_content = f"{fallback['risk_factors']}\n{fallback['mda']}"
            content_hash = compute_content_hash(combined_content)

            record = FilingRecord(
                ticker=ticker,
                cik=fallback["cik"],
                form_type="10-K",
                filing_date=filing_date,
                accession_number=fallback["accession"],
                period_end=filing_date,
                filed_at=filing_date,
                content_hash=content_hash,
                risk_factors_text=fallback["risk_factors"],
                mda_text=fallback["mda"],
                risk_factors_extraction=SectionExtraction(
                    section_name="Item 1A: Risk Factors",
                    content=fallback["risk_factors"],
                    char_start=0,
                    char_end=len(fallback["risk_factors"]),
                    extraction_status=ComponentStatus.OFFLINE_FALLBACK,
                    parser_version="1.0.0",
                    warnings=("Demo synthetic data",),
                ),
                mda_extraction=SectionExtraction(
                    section_name="Item 7: MD&A",
                    content=fallback["mda"],
                    char_start=0,
                    char_end=len(fallback["mda"]),
                    extraction_status=ComponentStatus.OFFLINE_FALLBACK,
                    parser_version="1.0.0",
                    warnings=("Demo synthetic data",),
                ),
                market_metrics=market_metrics,
                is_synthetic_fallback=True,
                execution_mode=ExecutionMode.DEMO,
            )

            # Cache demo separately with explicit DEMO marker
            demo_cache_file = CACHE_DIR / f"{ticker}_DEMO_10k.json"
            demo_cache_file.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")

            return record

    def load_filing_by_accession(self, ticker: str, accession: str) -> Optional[FilingRecord]:
        """
        Load a specific cached filing by accession number.
        Returns None if not found.
        """
        ticker = ticker.upper().strip()
        cache_file = CACHE_DIR / f"{ticker}_{accession}_10k.json"
        
        if not cache_file.exists():
            logger.info("Cached filing not found: %s", cache_file)
            return None
        
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            record = FilingRecord.from_dict(data)
            logger.info("Loaded cached filing for %s accession %s", ticker, accession)
            return record
        except Exception as exc:
            logger.warning("Failed to load cached filing %s: %s", cache_file, exc)
            return None

    def load_historical_panel(self, ticker: str) -> pd.DataFrame:
        """
        Loads historical panel from actual cached filings.
        Returns empty DataFrame if no historical filings available (no synthetic data).
        """
        ticker = ticker.upper().strip()

        # Find all cached filings for this ticker
        cached_files = list(CACHE_DIR.glob(f"{ticker}_*_10k.json"))
        cached_files = [f for f in cached_files if "DEMO" not in f.name]

        if not cached_files:
            logger.info("No historical live filings found for %s; returning empty panel", ticker)
            return pd.DataFrame(columns=[
                "ticker", "filing_date", "quarter", "optimism_index",
                "post_7d_volatility_pct", "post_7d_return_pct", "accession"
            ])

        rows = []
        for cache_file in sorted(cached_files):
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                record = FilingRecord.from_dict(data)

                # Only include if we have actual sentiment data (would need to be stored/computed)
                # For now, return what we have from the filing
                row = {
                    "ticker": record.ticker,
                    "filing_date": record.filing_date,
                    "quarter": f"Q{(int(record.filing_date[5:7]) - 1) // 3 + 1}",
                    "optimism_index": None,  # Would need historical sentiment
                    "post_7d_volatility_pct": record.market_metrics.post_7d_volatility_pct if record.market_metrics else None,
                    "post_7d_return_pct": record.market_metrics.post_7d_return_pct if record.market_metrics else None,
                    "accession": record.accession_number,
                }
                rows.append(row)
            except Exception as exc:
                logger.warning("Failed to load historical panel from %s: %s", cache_file, exc)

        if not rows:
            return pd.DataFrame(columns=[
                "ticker", "filing_date", "quarter", "optimism_index",
                "post_7d_volatility_pct", "post_7d_return_pct", "accession"
            ])

        df = pd.DataFrame(rows)
        df = df.sort_values("filing_date").reset_index(drop=True)
        return df