"""
audit_ingestion_v04.2.2/audit_ingestion/router.py
Pipeline orchestrator with per-stage timing and throttle semaphores.

Throttle caps (conservative):
  AI_SEMAPHORE:  2 concurrent canonical AI calls
  OCR_SEMAPHORE: 2 concurrent OCR operations

Per-stage timing is collected and stored in ExtractionMeta.
"""
from __future__ import annotations
import logging
import threading
import time
from pathlib import Path
from typing import Optional
from .models import AuditEvidence, IngestionResult, ExtractionMeta, Flag
from .extractor import extract
from .normalizers import normalize_evidence

logger = logging.getLogger(__name__)
ROUTER_BUILD = "v04.6-schemafix-3"

# ── Throttle semaphores ───────────────────────────────────────────────────────
# Shared across all threads in the process — conservative caps for stability
_AI_SEMAPHORE  = threading.Semaphore(2)   # max 2 concurrent canonical AI calls
_OCR_SEMAPHORE = threading.Semaphore(2)   # max 2 concurrent OCR operations


def ingest_one(
    path: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    mode: str = "fast",
    allow_rescue: bool = False,
) -> IngestionResult:
    """
    Ingest one document with per-stage timing and throttle semaphores.

    Args:
        path:         File path.
        api_key:      OpenAI API key.
        model:        Model override (defaults to CANONICAL_MODEL in provider).
        mode:         "fast" or "deep".
        allow_rescue: If True, attempt gpt-5.4-pro freeform rescue on worst pages.
                      Never used for canonical structured JSON.
    """
    input_p = Path(path)
    engine_chain: list[str] = []
    errors: list[str] = []
    stage_timings: dict[str, float] = {}

    if not input_p.exists():
        return IngestionResult(
            status="failed",
            errors=["File not found"],
            evidence=AuditEvidence(
                source_file=input_p.name,
                flags=[Flag(type="file_not_found",
                            description="File not found", severity="critical")]
            ),
        )

    # Initialize provider
    provider = None
    if api_key:
        try:
            from .providers import get_provider
            provider = get_provider("openai", api_key=api_key, model=model)
        except Exception as e:
            errors.append(f"Provider init failed: {e}")
            logger.error(f"Provider init: {e}")
    else:
        errors.append("No API key — extraction only, no canonical analysis")

    # Stage 1: Extract (respects OCR semaphore internally via extractor)
    t0 = time.perf_counter()
    try:
        # Pass semaphore to extractor for OCR throttling
        parsed_doc = extract(path, provider=provider, mode=mode,
                             ocr_semaphore=_OCR_SEMAPHORE)
        stage_timings["extraction"] = round(time.perf_counter() - t0, 3)
        engine_chain.extend(parsed_doc.extraction_chain)
        if parsed_doc.errors:
            errors.extend(parsed_doc.errors)
    except Exception as e:
        stage_timings["extraction"] = round(time.perf_counter() - t0, 3)
        errors.append(f"Extraction failed: {e}")
        return IngestionResult(
            status="failed",
            errors=errors,
            evidence=AuditEvidence(
                source_file=input_p.name,
                flags=[Flag(type="extraction_error",
                            description=str(e), severity="critical")]
            ),
        )

    meta = ExtractionMeta(
        primary_extractor=parsed_doc.primary_extractor,
        pages_processed=parsed_doc.page_count,
        weak_pages_count=len(parsed_doc.weak_pages),
        ocr_pages_count=len(parsed_doc.ocr_pages),
        vision_pages_count=len(parsed_doc.vision_pages),
        total_chars=len(parsed_doc.full_text),
        overall_confidence=parsed_doc.confidence,
        needs_human_review=not parsed_doc.is_sufficient,
        warnings=parsed_doc.warnings,
        errors=errors,
    )

    # Stage 2: Canonical AI extraction (throttled)
    if provider is not None and parsed_doc.full_text:
        t1 = time.perf_counter()
        with _AI_SEMAPHORE:
            try:
                from .canonical import extract_canonical
                evidence = extract_canonical(parsed_doc, provider)
                engine_chain.append("canonical_ai")
            except Exception as e:
                errors.append(f"Canonical extraction failed: {e}")
                logger.error(f"Canonical: {e}")
                evidence = AuditEvidence(
                    source_file=input_p.name,
                    raw_text=parsed_doc.full_text,
                    tables=[t if isinstance(t, dict) else t.model_dump()
                            for t in parsed_doc.tables],
                    extraction_meta=meta,
                    flags=[Flag(type="canonical_failed",
                                description=str(e), severity="critical")]
                )
                engine_chain.append("canonical_failed")
        stage_timings["canonical_ai"] = round(time.perf_counter() - t1, 3)


        # Stage 3: Optional gpt-5.4-pro visual rescue on worst pages
        # Uses page IMAGES via vision model — not weak text re-interpretation
        # Selects pages by LOWEST char count (truly worst), not lowest page number
        # Guarded by _AI_SEMAPHORE — respects global AI concurrency cap
        # NEVER used for canonical structured JSON (always gpt-5.4 only)
        if allow_rescue and parsed_doc.weak_pages and provider is not None:
            worst_pages = sorted(
                [p for p in parsed_doc.pages if p.page_number in parsed_doc.weak_pages],
                key=lambda p: p.char_count   # lowest char count first = truly worst
            )[:2]

            if worst_pages:
                t2 = time.perf_counter()
                try:
                    from .providers.openai_provider import RESCUE_MODEL
                    rescue_texts = []

                    with _AI_SEMAPHORE:
                        for pg in worst_pages:
                            img = render_page_image_cached(path, pg.page_number - 1, dpi=200)
                            if not img:
                                continue
                            rescued = provider.extract_text_from_page_images(
                                images=[img],
                                prompt=(
                                    f"Read page {pg.page_number} of this document image. "
                                    f"Extract all audit-relevant facts you can find. "
                                    f"Return plain text only."
                                ),
                                model=RESCUE_MODEL,
                            )
                            if rescued and rescued.strip():
                                rescue_texts.append(
                                    f"[Rescued page {pg.page_number}]\n{rescued.strip()}"
                                )
                                engine_chain.append(f"rescue_p{pg.page_number}")

                    if rescue_texts:
                        rescue_combined = "\n\n".join(rescue_texts)
                        evidence.flags.append(Flag(
                            type="rescue_applied",
                            description=(
                                f"gpt-5.4-pro visual rescue applied to {len(rescue_texts)} page(s). "
                                f"Review rescued content in document_specific."
                            ),
                            severity="info",
                        ))
                        evidence.document_specific["rescued_page_text"] = rescue_combined

                    stage_timings["rescue"] = round(time.perf_counter() - t2, 3)
                except Exception as e:
                    logger.warning(f"Rescue pass failed: {e}")
                    stage_timings["rescue"] = round(time.perf_counter() - t2, 3)
                except Exception as e:
                    logger.warning(f"Rescue pass failed: {e}")
                    stage_timings["rescue"] = round(time.perf_counter() - t2, 3)

    else:
        flag_type = "no_ai" if not api_key else "no_text"
        flag_desc = ("No API key — canonical extraction skipped"
                     if not api_key else "No text extracted")
        evidence = AuditEvidence(
            source_file=input_p.name,
            raw_text=parsed_doc.full_text,
            tables=[t if isinstance(t, dict) else t.model_dump()
                    for t in parsed_doc.tables],
            extraction_meta=meta,
            flags=[Flag(type=flag_type, description=flag_desc, severity="warning")]
        )
        engine_chain.append("extraction_only")
        stage_timings["canonical_ai"] = 0.0

    # Stage 4: Normalize
    t3 = time.perf_counter()
    try:
        evidence = normalize_evidence(evidence)
        engine_chain.append("normalized")
    except Exception as e:
        logger.warning(f"Normalization failed: {e}")
    stage_timings["normalization"] = round(time.perf_counter() - t3, 3)

    # Stage 5: Score
    score = _score(evidence)
    evidence.extraction_meta.overall_confidence = score
    evidence.extraction_meta.needs_human_review = score < 0.70

    # Store stage timings in document_specific for UI display
    evidence.document_specific["_stage_timings"] = stage_timings
    stage_timings["total"] = round(
        sum(v for v in stage_timings.values()), 3
    )

    # If AI was unavailable or failed but we have usable extracted text, mark PARTIAL not FAILED
    # A file with good text is still useful — auditor can read raw text even without canonical JSON
    ai_unavailable = any(f.type in ("canonical_failed", "no_ai") for f in evidence.flags)
    has_text = (evidence.extraction_meta.total_chars or 0) >= 200

    if ai_unavailable and has_text and score < 0.30:
        score = 0.30  # Floor to PARTIAL when extraction succeeded but AI was unavailable
        evidence.extraction_meta.overall_confidence = score

    status = "success" if score >= 0.70 else ("partial" if score >= 0.30 else "failed")

    return IngestionResult(
        evidence=evidence,
        status=status,
        errors=errors,
        engine_chain=engine_chain,
    )


def _score(ev: AuditEvidence) -> float:
    s = 0.0
    if ev.audit_overview and ev.audit_overview.summary:
        s += 0.20
    s += 0.07 if ev.amounts  else 0
    s += 0.07 if ev.parties  else 0
    s += 0.06 if ev.dates    else 0
    s += 0.05 if ev.facts    else 0
    if ev.claims:
        s += min(0.15, len(ev.claims) * 0.05)
    all_items = (
        [(a.provenance, a.value) for a in ev.amounts] +
        [(p.provenance, p.name) for p in ev.parties] +
        [(d.provenance, d.value) for d in ev.dates]
    )
    if all_items:
        with_prov = sum(1 for prov, _ in all_items if prov and prov.confidence > 0.5)
        s += 0.20 * (with_prov / len(all_items))
    lk = ev.link_keys
    if any([lk.party_names, lk.document_numbers, lk.invoice_numbers,
            lk.agreement_numbers, lk.recurring_amounts]):
        s += 0.10
    if ev.extraction_meta.total_chars >= 500:
        s += 0.10
    elif ev.extraction_meta.total_chars >= 200:
        s += 0.05
    return round(min(s, 1.0), 3)
