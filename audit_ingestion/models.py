"""
audit_ingestion_v04.2/audit_ingestion/models.py
Canonical audit evidence schema — Pydantic models.

Two layers:
1. ParsedDocument — page-aware raw extraction output
2. AuditEvidence  — canonical audit evidence output

Every document goes through both layers.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, Any, Literal
from enum import Enum


# ── Document Family ───────────────────────────────────────────────────────────

class DocumentFamily(str, Enum):
    CONTRACT       = "contract_agreement"
    INVOICE        = "invoice_receipt"
    PAYMENT        = "payment_proof"
    BANK           = "bank_cash_activity"
    PAYROLL        = "payroll_support"
    ACCOUNTING     = "accounting_report"
    GOVERNANCE     = "governance_approval"
    GRANT          = "grant_donor_funding"
    TAX_REG        = "tax_regulatory"
    CORRESPONDENCE = "correspondence"
    SCHEDULE       = "schedule_listing"
    OTHER          = "other"


# ── Layer 1: Page-Aware Parsed Document ───────────────────────────────────────

class ParsedPage(BaseModel):
    """Text extracted from one PDF page."""
    page_number:  int
    text:         str = ""
    char_count:   int = 0
    extractor:    str = "none"     # pdfplumber, pypdf2, extractous, ocr, vision
    confidence:   float = 0.0
    image_used:   bool = False
    warnings:     list[str] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        if not self.char_count:
            self.char_count = len(self.text)


class ParsedTable(BaseModel):
    """Table extracted from a document page."""
    page_number:  int
    table_index:  int = 0
    headers:      list[str] = Field(default_factory=list)
    rows:         list[dict[str, Any]] = Field(default_factory=list)
    row_count:    int = 0
    extractor:    str = "unknown"


class ParsedDocument(BaseModel):
    """
    Page-aware extraction output — the foundation for canonical extraction.
    Every page has its own text record and provenance.
    """
    source_file:      str
    file_hash:        Optional[str] = None   # MD5 of file content — used for cache keying
    mime_type:        Optional[str] = None
    full_text:        str = ""           # Assembled from page texts
    page_count:       int = 0
    pages:            list[ParsedPage]   = Field(default_factory=list)
    tables:           list[ParsedTable]  = Field(default_factory=list)
    extraction_chain: list[str]          = Field(default_factory=list)
    primary_extractor: str = "none"
    confidence:       float = 0.0
    weak_pages:       list[int]          = Field(default_factory=list)  # Pages below threshold
    ocr_pages:        list[int]          = Field(default_factory=list)  # Pages rescued by OCR
    vision_pages:     list[int]          = Field(default_factory=list)  # Pages rescued by vision
    warnings:         list[str]          = Field(default_factory=list)
    errors:           list[str]          = Field(default_factory=list)

    @property
    def chars_per_page(self) -> float:
        if not self.page_count:
            return 0.0
        return len(self.full_text) / self.page_count

    @property
    def is_sufficient(self) -> bool:
        return len(self.full_text) >= 300 and self.chars_per_page >= 150


# ── Layer 2: Canonical Audit Evidence ────────────────────────────────────────

class Provenance(BaseModel):
    """Source evidence for any extracted item — mandatory for material facts."""
    page:       Optional[int]   = None
    quote:      Optional[str]   = None   # ≤20 word verbatim excerpt
    confidence: float           = 0.0


class Party(BaseModel):
    role:       str                      # lessor, vendor, grantor, payer, client, etc.
    name:       str
    normalized: str                      # UPPERCASE no-punctuation for matching
    provenance: Optional[Provenance] = None


class Amount(BaseModel):
    type:       str                      # monthly_fixed_charge, total_award, etc.
    value:      float
    currency:   str = "USD"
    provenance: Optional[Provenance] = None


class DateItem(BaseModel):
    type:       str                      # effective_date, invoice_date, period_start, etc.
    value:      str                      # YYYY-MM-DD
    provenance: Optional[Provenance] = None


class Identifier(BaseModel):
    type:       str                      # invoice_number, schedule_number, grant_number, etc.
    value:      str
    provenance: Optional[Provenance] = None


class AssetItem(BaseModel):
    type:        str                     # vehicle, equipment, property, program, etc.
    description: str
    value:       Optional[float] = None
    provenance:  Optional[Provenance] = None


class Fact(BaseModel):
    """Atomic extracted fact — drives matching. Must have provenance."""
    label:      str                      # snake_case: term_months, mileage_rate, etc.
    value:      Any
    provenance: Optional[Provenance] = None


class Claim(BaseModel):
    """Auditor-readable interpretation built from facts. Must cite source facts."""
    statement:         str
    audit_area:        str               # leases, expenses, revenue, etc.
    basis_fact_labels: list[str] = Field(default_factory=list)
    provenance:        Optional[Provenance] = None


class Flag(BaseModel):
    """Audit exception, risk, or attention item."""
    type:        str
    description: str
    severity:    Literal["info", "warning", "critical"] = "info"


class AuditPeriod(BaseModel):
    effective_date: Optional[str] = None
    start:          Optional[str] = None
    end:            Optional[str] = None
    term_months:    Optional[int] = None


class AuditOverview(BaseModel):
    summary:       str
    audit_areas:   list[str] = Field(default_factory=list)
    assertions:    list[str] = Field(default_factory=list)
    period:        Optional[AuditPeriod] = None
    match_targets: list[str] = Field(default_factory=list)


class LinkKeys(BaseModel):
    """Normalized keys for cross-document matching."""
    party_names:       list[str]   = Field(default_factory=list)
    document_numbers:  list[str]   = Field(default_factory=list)
    agreement_numbers: list[str]   = Field(default_factory=list)
    invoice_numbers:   list[str]   = Field(default_factory=list)
    asset_descriptions:list[str]   = Field(default_factory=list)
    recurring_amounts: list[float] = Field(default_factory=list)
    key_dates:         list[str]   = Field(default_factory=list)
    other_ids:         list[str]   = Field(default_factory=list)


class ExtractionMeta(BaseModel):
    primary_extractor:    str = "none"
    pages_processed:      int = 0
    weak_pages_count:     int = 0
    ocr_pages_count:      int = 0
    vision_pages_count:   int = 0
    total_chars:          int = 0
    overall_confidence:   float = 0.0
    needs_human_review:   bool = True
    canonical_validated:  bool = False
    canonical_retried:    bool = False
    warnings:             list[str] = Field(default_factory=list)
    errors:               list[str] = Field(default_factory=list)


class AuditEvidence(BaseModel):
    """
    Canonical audit evidence object.
    One per document. Always the same shape.
    Works for any document type.
    """
    source_file:      str
    family:           DocumentFamily = DocumentFamily.OTHER
    subtype:          Optional[str] = None
    title:            Optional[str] = None
    audit_overview:   Optional[AuditOverview] = None
    parties:          list[Party]      = Field(default_factory=list)
    amounts:          list[Amount]     = Field(default_factory=list)
    dates:            list[DateItem]   = Field(default_factory=list)
    identifiers:      list[Identifier] = Field(default_factory=list)
    assets:           list[AssetItem]  = Field(default_factory=list)
    facts:            list[Fact]       = Field(default_factory=list)
    claims:           list[Claim]      = Field(default_factory=list)
    flags:            list[Flag]       = Field(default_factory=list)
    link_keys:        LinkKeys         = Field(default_factory=LinkKeys)
    document_specific:dict[str, Any]   = Field(default_factory=dict)
    raw_text:         Optional[str]    = None
    tables:           list[dict]       = Field(default_factory=list)
    extraction_meta:  ExtractionMeta   = Field(
        default_factory=lambda: ExtractionMeta(primary_extractor="none")
    )


class IngestionResult(BaseModel):
    evidence:     Optional[AuditEvidence] = None
    status:       Literal["success", "partial", "failed"] = "partial"
    errors:       list[str] = Field(default_factory=list)
    engine_chain: list[str] = Field(default_factory=list)
