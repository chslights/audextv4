"""
audit_ingestion_v04/ingest_app.py
Audit Ingestion Pipeline v04.2 — Streamlit UI

Canonical audit evidence view with extraction diagnostics.
OpenAI only. One model selector. Full provenance display.
"""
import streamlit as st
import json
import sys
import tempfile
import shutil
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(
    page_title="Audit Ingestion v04.2",
    page_icon="📋",
    layout="wide",
)

st.markdown("""
<style>
.page-header {
    font-size: 1.4rem; font-weight: 800; color: #1A335C;
    margin-bottom: 0;
}
.section-title {
    font-size: 0.95rem; font-weight: 700; color: #1A335C;
    border-bottom: 2px solid #1A335C;
    padding-bottom: 3px; margin: 16px 0 10px 0;
}
.audit-area-tag {
    background: #dbeafe; color: #1e40af;
    padding: 2px 8px; border-radius: 3px;
    font-size: 0.75rem; font-weight: 600;
    display: inline-block; margin: 2px;
}
.assertion-tag {
    background: #dcfce7; color: #166534;
    padding: 2px 8px; border-radius: 3px;
    font-size: 0.75rem; font-weight: 600;
    display: inline-block; margin: 2px;
}
.match-target-tag {
    background: #f3e8ff; color: #6b21a8;
    padding: 2px 8px; border-radius: 3px;
    font-size: 0.75rem; font-weight: 600;
    display: inline-block; margin: 2px;
}
.claim-box {
    background: #f0fdf4; border-left: 3px solid #16a34a;
    padding: 8px 12px; margin: 6px 0; border-radius: 0 4px 4px 0;
}
.flag-info     { background: #eff6ff; border-left: 3px solid #3b82f6; padding: 8px 12px; margin: 4px 0; border-radius: 0 4px 4px 0; }
.flag-warning  { background: #fffbeb; border-left: 3px solid #f59e0b; padding: 8px 12px; margin: 4px 0; border-radius: 0 4px 4px 0; }
.flag-critical { background: #fef2f2; border-left: 3px solid #dc2626; padding: 8px 12px; margin: 4px 0; border-radius: 0 4px 4px 0; }
.diag-row { font-size: 0.82rem; color: #374151; }
.prov-quote { font-style: italic; color: #6b7280; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)


def conf_badge(c: float) -> str:
    if c >= 0.80: color = "#16a34a"
    elif c >= 0.50: color = "#d97706"
    else: color = "#dc2626"
    return f'<span style="color:{color};font-weight:700">{c:.2f}</span>'


def _process_one(args):
    """Worker function for concurrent file processing."""
    import time
    path, api_key, mode, model, allow_rescue = args
    t0 = time.time()
    try:
        from audit_ingestion.router import ingest_one
        result = ingest_one(path, api_key=api_key, model=model, mode=mode,
                            allow_rescue=allow_rescue)
        elapsed = round(time.time() - t0, 2)
        return result, elapsed, None
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        return None, elapsed, str(e)


def run_pipeline(uploaded_files, api_key, mode="fast", allow_rescue=False):
    """
    Run pipeline with concurrent file workers.
    File workers: 4 (conservative)
    AI concurrency implicitly capped by worker count.
    """
    import time
    import concurrent.futures
    from audit_ingestion.providers.openai_provider import DEFAULT_MODEL
    from audit_ingestion.models import IngestionResult, AuditEvidence, Flag

    FILE_WORKERS = min(4, len(uploaded_files))
    batch_start = time.time()

    progress    = st.progress(0)
    status_text = st.empty()
    stage_area  = st.empty()

    tmpdir = tempfile.mkdtemp(prefix="audit_v042_")
    Path(tmpdir, ".tmp").mkdir(exist_ok=True)

    # Write all files first
    file_paths = []
    for uf in uploaded_files:
        tmp_path = Path(tmpdir) / uf.name
        with open(tmp_path, "wb") as f:
            f.write(uf.read())
        file_paths.append(str(tmp_path))

    results = [None] * len(file_paths)
    timings: dict[str, float] = {}
    completed = 0

    try:
        args_list = [
            (path, api_key, mode, DEFAULT_MODEL, allow_rescue)
            for path in file_paths
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=FILE_WORKERS) as executor:
            future_to_idx = {
                executor.submit(_process_one, args): i
                for i, args in enumerate(args_list)
            }

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                fname = Path(file_paths[idx]).name
                completed += 1
                progress.progress(completed / len(file_paths))

                try:
                    result, elapsed, err = future.result()
                    timings[fname] = elapsed
                    if result is not None:
                        results[idx] = result
                        stage = result.engine_chain[-1] if result.engine_chain else "?"
                        status_text.text(
                            f"✅ {fname} — {elapsed}s | chain: {' → '.join(result.engine_chain)}"
                        )
                    else:
                        results[idx] = IngestionResult(
                            status="failed",
                            errors=[err or "Unknown error"],
                            evidence=AuditEvidence(
                                source_file=fname,
                                flags=[Flag(type="fatal_error",
                                           description=err or "Unknown", severity="critical")]
                            ),
                        )
                        status_text.text(f"❌ {fname} failed — {elapsed}s")
                except Exception as e:
                    results[idx] = IngestionResult(
                        status="failed",
                        errors=[str(e)],
                        evidence=AuditEvidence(
                            source_file=fname,
                            flags=[Flag(type="fatal_error",
                                       description=str(e), severity="critical")]
                        ),
                    )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        progress.empty()
        status_text.empty()

    batch_elapsed = round(time.time() - batch_start, 2)
    st.caption(
        f"Batch complete — {len(results)} files in {batch_elapsed}s "
        f"({FILE_WORKERS} workers | mode: {mode})"
    )

    return results, timings


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    # API Key
    key_file = Path("openai_key.txt")
    default_key = key_file.read_text().strip() if key_file.exists() else ""
    api_key = st.text_input(
        "OpenAI API Key",
        value=default_key, type="password", placeholder="sk-...",
    )
    if api_key:
        st.success("✅ Key ready")
    else:
        st.warning("⚠️ API key required for extraction")

    # Processing Mode
    processing_mode = st.radio(
        "Processing Mode",
        ["Fast Review", "Deep Extraction"],
        index=0,
        help=(
            "Fast Review: pdfplumber + PyPDF2 + auto-escalation. Quick.\n"
            "Deep Extraction: adds OCR + vision on weak pages."
        ),
    )
    mode = "fast" if processing_mode == "Fast Review" else "deep"

    allow_rescue = st.checkbox(
        "Allow gpt-5.4-pro rescue on worst page(s)",
        value=False,
        help=(
            "Off by default. When enabled, gpt-5.4-pro may be used for freeform "
            "rescue on pages that canonical extraction cannot read. "
            "NOT used for canonical structured JSON."
        ),
    )
    st.caption(f"Model: `gpt-5.4` | Mode: `{mode}`")

    st.markdown("---")
    st.markdown("### 📋 v04.2 Architecture")
    st.markdown("""
1. **Page-aware extraction**
   pdfplumber → PyPDF2 → extractous → OCR → vision
2. **Canonical AI pass**
   Single structured JSON extraction
3. **Normalization**
   Parties, dates, amounts, link keys
4. **Audit evidence object**
   Facts + Claims + Flags + Link Keys
    """)
    st.caption("Audit Ingestion Pipeline v04.2")


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-header">📋 Audit Ingestion Pipeline</div>', unsafe_allow_html=True)
st.markdown("Upload any audit document — lease, invoice, grant, bank statement, minutes, payroll, or any other.")
st.markdown("---")

uploaded_files = st.file_uploader(
    "Drop documents here",
    accept_multiple_files=True,
    type=["pdf", "csv", "xlsx", "xls", "txt", "docx"],
    label_visibility="visible",
)

c1, c2 = st.columns([2, 1])
with c1:
    run_btn = st.button("▶ Run Pipeline", type="primary",
                        disabled=not uploaded_files or not api_key)
with c2:
    if st.button("🗑 Clear"):
        st.session_state.pop("v042_results", None)
        st.rerun()

if not api_key and uploaded_files:
    st.warning("Enter your OpenAI API key in the sidebar to run extraction.")

if run_btn and uploaded_files and api_key:
    with st.spinner("Running page-aware extraction and canonical AI analysis..."):
        results, timings = run_pipeline(uploaded_files, api_key, mode, allow_rescue)
    st.session_state["v042_results"] = [r.model_dump() for r in results]
    st.session_state["v042_timings"] = timings
    st.rerun()


# ── Results ───────────────────────────────────────────────────────────────────
if "v042_results" not in st.session_state:
    st.stop()

raw_results = st.session_state["v042_results"]

# Metrics
total   = len(raw_results)
success = sum(1 for r in raw_results if r["status"] == "success")
partial = sum(1 for r in raw_results if r["status"] == "partial")
failed  = sum(1 for r in raw_results if r["status"] == "failed")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total", total)
m2.metric("✅ Success", success)
m3.metric("⚠️ Partial", partial)
m4.metric("❌ Failed", failed)

st.markdown("---")

# Summary table
st.markdown('<div class="section-title">Document Summary</div>', unsafe_allow_html=True)

from audit_ingestion.legacy import canonical_summary_row
from audit_ingestion.models import AuditEvidence

summary_rows = []
for r in raw_results:
    ev_data = r.get("evidence") or {}
    try:
        ev = AuditEvidence(**ev_data)
        row = canonical_summary_row(ev)
        row["status"] = r["status"].upper()
        summary_rows.append(row)
    except Exception:
        summary_rows.append({
            "file": ev_data.get("source_file", "?"),
            "status": r["status"].upper(),
            "family": "?", "summary": "Parse error",
            "primary_party": "—", "primary_amount": "—",
            "audit_areas": "—", "confidence": "0.00",
            "extractor": "—", "chars": 0, "needs_review": True,
        })

df = pd.DataFrame(summary_rows)

def highlight(row):
    s = row.get("status", "")
    if s == "SUCCESS":   return ["background-color: #f0fdf4"] * len(row)
    elif s == "PARTIAL": return ["background-color: #fffbeb"] * len(row)
    return ["background-color: #fef2f2"] * len(row)

timings = st.session_state.get("v042_timings", {})
for row in summary_rows:
    fname = row.get("file", "")
    row["time_s"] = f"{timings.get(fname, 0):.1f}s"

display_cols = ["file", "status", "family", "subtype", "primary_party",
                "primary_amount", "audit_areas", "confidence", "extractor",
                "chars", "time_s", "needs_review"]
display_cols = [c for c in display_cols if c in df.columns]

st.dataframe(df[display_cols].style.apply(highlight, axis=1),
             use_container_width=True, hide_index=True)

st.download_button("⬇️ Export CSV",
                   data=df.to_csv(index=False),
                   file_name="audit_evidence_v042.csv",
                   mime="text/csv")

st.markdown("---")

# File detail selector
st.markdown('<div class="section-title">Document Detail</div>', unsafe_allow_html=True)
file_names = [r.get("evidence", {}).get("source_file", f"File {i}")
              for i, r in enumerate(raw_results)]
selected = st.selectbox("Select document to inspect", file_names)

r = next((x for x in raw_results
          if x.get("evidence", {}).get("source_file") == selected), None)
if not r:
    st.stop()

ev_data = r.get("evidence") or {}
try:
    ev = AuditEvidence(**ev_data)
except Exception as e:
    st.error(f"Could not parse evidence: {e}")
    st.stop()

meta = ev.extraction_meta
overview = ev.audit_overview

# ── Extraction Diagnostics ────────────────────────────────────────────────────
with st.expander("🔬 Extraction Diagnostics", expanded=True):
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Extractor", meta.primary_extractor)
    d2.metric("Pages", meta.pages_processed)
    d3.metric("Total Chars", f"{meta.total_chars:,}")
    d4.metric("Confidence", f"{meta.overall_confidence:.2f}")

    d5, d6, d7, d8 = st.columns(4)
    d5.metric("Weak Pages", meta.weak_pages_count)
    d6.metric("OCR Pages", meta.ocr_pages_count)
    d7.metric("Vision Pages", meta.vision_pages_count)
    d8.metric("Needs Review", "Yes" if meta.needs_human_review else "No")

    if meta.warnings:
        st.warning("**Warnings:** " + " | ".join(meta.warnings))
    if meta.errors:
        st.error("**Errors:** " + " | ".join(meta.errors))

    # Engine chain
    chain = r.get("engine_chain", [])
    if chain:
        st.markdown(f"**Engine Chain:** `{' → '.join(chain)}`")

    # Per-stage timing
    doc_specific = ev_data.get("document_specific") or {}
    stage_timings = doc_specific.get("_stage_timings")
    if stage_timings:
        timing_cols = st.columns(len(stage_timings))
        for i, (stage, t) in enumerate(stage_timings.items()):
            timing_cols[i].metric(stage.replace("_", " ").title(), f"{t:.2f}s")

# ── Section 1: Auditor Snapshot ───────────────────────────────────────────────
if overview:
    st.markdown('<div class="section-title">🔍 Auditor Snapshot</div>', unsafe_allow_html=True)

    family = ev.family.value.replace("_", " ").title()
    subtype = ev.subtype or "—"
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Document Family:** `{family}`")
    c2.markdown(f"**Subtype:** `{subtype}`")
    c3.markdown(f"**Title:** {ev.title or '—'}")

    st.markdown(f"**Summary:** {overview.summary}")

    if overview.audit_areas:
        tags = " ".join(f'<span class="audit-area-tag">{a}</span>'
                        for a in overview.audit_areas)
        st.markdown(f"**Audit Areas:** {tags}", unsafe_allow_html=True)

    if overview.assertions:
        tags = " ".join(f'<span class="assertion-tag">{a}</span>'
                        for a in overview.assertions)
        st.markdown(f"**Assertions:** {tags}", unsafe_allow_html=True)

    if overview.period:
        p = overview.period
        period_parts = []
        if p.effective_date: period_parts.append(f"Effective: {p.effective_date}")
        if p.start:          period_parts.append(f"Start: {p.start}")
        if p.end:            period_parts.append(f"End: {p.end}")
        if p.term_months:    period_parts.append(f"Term: {p.term_months} months")
        if period_parts:
            st.markdown(f"**Period:** {' | '.join(period_parts)}")

    if overview.match_targets:
        tags = " ".join(f'<span class="match-target-tag">{t}</span>'
                        for t in overview.match_targets)
        st.markdown(f"**Match Targets:** {tags}", unsafe_allow_html=True)

# ── Section 2: Key Audit Facts ────────────────────────────────────────────────
st.markdown('<div class="section-title">📊 Key Audit Facts</div>', unsafe_allow_html=True)

col_l, col_r = st.columns(2)

with col_l:
    if ev.parties:
        st.markdown("**Parties**")
        party_rows = []
        for p in ev.parties:
            prov = p.provenance
            party_rows.append({
                "Role": p.role,
                "Name": p.name,
                "Page": prov.page if prov else "—",
                "Quote": prov.quote if prov else "—",
                "Conf": f"{prov.confidence:.2f}" if prov else "—",
            })
        st.dataframe(pd.DataFrame(party_rows), use_container_width=True, hide_index=True)

    if ev.amounts:
        st.markdown("**Amounts**")
        amt_rows = []
        for a in ev.amounts:
            prov = a.provenance
            amt_rows.append({
                "Type": a.type,
                "Amount": f"${a.value:,.2f}",
                "Page": prov.page if prov else "—",
                "Quote": prov.quote if prov else "—",
                "Conf": f"{prov.confidence:.2f}" if prov else "—",
            })
        st.dataframe(pd.DataFrame(amt_rows), use_container_width=True, hide_index=True)

with col_r:
    if ev.dates:
        st.markdown("**Dates**")
        date_rows = []
        for d in ev.dates:
            prov = d.provenance
            date_rows.append({
                "Type": d.type,
                "Value": d.value,
                "Page": prov.page if prov else "—",
                "Quote": prov.quote if prov else "—",
            })
        st.dataframe(pd.DataFrame(date_rows), use_container_width=True, hide_index=True)

    if ev.identifiers:
        st.markdown("**Identifiers**")
        id_rows = []
        for ident in ev.identifiers:
            prov = ident.provenance
            id_rows.append({
                "Type": ident.type,
                "Value": ident.value,
                "Page": prov.page if prov else "—",
                "Quote": prov.quote if prov else "—",
            })
        st.dataframe(pd.DataFrame(id_rows), use_container_width=True, hide_index=True)

if ev.assets:
    st.markdown("**Assets / Items**")
    asset_rows = [{
        "Type": a.type,
        "Description": a.description,
        "Value": f"${a.value:,.2f}" if a.value else "—",
        "Page": a.provenance.page if a.provenance else "—",
    } for a in ev.assets]
    st.dataframe(pd.DataFrame(asset_rows), use_container_width=True, hide_index=True)

if ev.facts:
    with st.expander(f"📌 Atomic Facts ({len(ev.facts)})", expanded=False):
        fact_rows = [{
            "Label": f.label,
            "Value": str(f.value),
            "Page": f.provenance.page if f.provenance else "—",
            "Quote": f.provenance.quote if f.provenance else "—",
            "Conf": f"{f.provenance.confidence:.2f}" if f.provenance else "—",
        } for f in ev.facts]
        st.dataframe(pd.DataFrame(fact_rows), use_container_width=True, hide_index=True)

# ── Section 3: Claims ─────────────────────────────────────────────────────────
if ev.claims:
    st.markdown('<div class="section-title">📝 Auditor Claims</div>', unsafe_allow_html=True)
    for c in ev.claims:
        prov = c.provenance
        quote_html = f'<br><span class="prov-quote">"{prov.quote}"</span>' if prov and prov.quote else ""
        conf_html = f" | Conf: {prov.confidence:.2f}" if prov else ""
        basis = f" | Based on: {', '.join(c.basis_fact_labels)}" if c.basis_fact_labels else ""
        st.markdown(
            f'<div class="claim-box">'
            f'<strong>{c.statement}</strong>'
            f'<br><small>Area: <strong>{c.audit_area}</strong>'
            f'{conf_html}{basis}'
            f'Page: {prov.page if prov and prov.page else "?"}'
            f'</small>{quote_html}'
            f'</div>',
            unsafe_allow_html=True
        )

# ── Section 4: Flags ──────────────────────────────────────────────────────────
if ev.flags:
    st.markdown('<div class="section-title">🚩 Flags & Exceptions</div>', unsafe_allow_html=True)
    for flag in ev.flags:
        severity = flag.severity
        st.markdown(
            f'<div class="flag-{severity}">'
            f'<strong>[{severity.upper()}] {flag.type}</strong><br>'
            f'{flag.description}'
            f'</div>',
            unsafe_allow_html=True
        )

# ── Section 5: Link Keys ──────────────────────────────────────────────────────
lk = ev.link_keys
has_links = any([lk.party_names, lk.document_numbers, lk.invoice_numbers,
                 lk.agreement_numbers, lk.recurring_amounts, lk.other_ids])
if has_links:
    with st.expander("🔗 Link Keys — Cross-Document Matching", expanded=False):
        st.caption("Normalized keys for matching against GL, AP, fixed assets, and other evidence.")
        lk_rows = []
        for field, values in lk.model_dump().items():
            if values:
                lk_rows.append({
                    "Key Type": field.replace("_", " ").title(),
                    "Values": ", ".join(str(v) for v in values)
                })
        if lk_rows:
            st.dataframe(pd.DataFrame(lk_rows), use_container_width=True, hide_index=True)

# ── Section 6: Document Specific ─────────────────────────────────────────────
if ev.document_specific:
    with st.expander("📄 Document-Specific Fields", expanded=False):
        st.json(ev.document_specific)

# ── Section 7: Tables ─────────────────────────────────────────────────────────
if ev.tables:
    with st.expander(f"📊 Extracted Tables ({len(ev.tables)})", expanded=False):
        for i, tbl in enumerate(ev.tables):
            page_n = tbl.get("page_number", tbl.get("page", "?"))
            st.markdown(f"**Table {i+1}** (page {page_n})")
            rows = tbl.get("rows") or []
            if rows:
                try:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                except Exception:
                    st.json(rows[:5])

# ── Section 8: Raw Text by Page ───────────────────────────────────────────────
if ev.raw_text:
    with st.expander("📝 Raw Extracted Text", expanded=False):
        st.text(ev.raw_text[:5000])

# ── Section 9: Full Canonical JSON ───────────────────────────────────────────
with st.expander("🔧 Full Canonical JSON", expanded=False):
    st.json(ev.model_dump())
