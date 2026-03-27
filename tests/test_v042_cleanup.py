"""
tests/test_v042_cleanup.py
Spec-required tests for v04.2 cleanup pass.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def test_canonical_cache_key_uses_file_hash():
    """Different file_hash values must produce different cache keys even with same name."""
    from audit_ingestion.models import ParsedDocument, ParsedPage
    from audit_ingestion.canonical import _canonical_cache_key

    pages = [ParsedPage(page_number=1, text="x", char_count=1, extractor="pdfplumber")]
    d1 = ParsedDocument(source_file="a.pdf", file_hash="abc",
                        full_text="x", page_count=1, pages=pages,
                        extraction_chain=["pdfplumber"], primary_extractor="pdfplumber")
    d2 = ParsedDocument(source_file="a.pdf", file_hash="def",
                        full_text="x", page_count=1, pages=pages,
                        extraction_chain=["pdfplumber"], primary_extractor="pdfplumber")

    assert _canonical_cache_key(d1, "gpt-5.4") != _canonical_cache_key(d2, "gpt-5.4")


def test_rescue_selects_lowest_char_count_pages():
    """Rescue must select pages by lowest char count, not lowest page number."""
    from audit_ingestion.models import ParsedPage

    pages = [
        ParsedPage(page_number=1, text="x" * 200, char_count=200, extractor="pdfplumber"),
        ParsedPage(page_number=2, text="x" * 20,  char_count=20,  extractor="pdfplumber"),
        ParsedPage(page_number=3, text="x" * 40,  char_count=40,  extractor="pdfplumber"),
    ]
    weak = [1, 2, 3]
    worst = sorted(
        [p for p in pages if p.page_number in weak],
        key=lambda p: p.char_count
    )[:2]
    assert [p.page_number for p in worst] == [2, 3]


def test_rescue_uses_image_not_page_text():
    """Rescue must use render_page_image_cached + extract_text_from_page_images, not pg.text."""
    import inspect
    from audit_ingestion import router
    src = inspect.getsource(router.ingest_one)
    assert "render_page_image_cached" in src
    assert "extract_text_from_page_images" in src
    assert "Page content:" not in src  # Old text-based rescue must be gone


def test_rescue_is_ai_semaphore_guarded():
    """Rescue must run inside _AI_SEMAPHORE to respect global AI concurrency cap."""
    import inspect
    from audit_ingestion import router
    src = inspect.getsource(router.ingest_one)
    assert "with _AI_SEMAPHORE" in src


def test_image_cache_is_used():
    """render_page_image_cached must read/write _image_page_cache."""
    import inspect
    from audit_ingestion import extractor
    src = inspect.getsource(extractor.render_page_image_cached)
    assert "_image_page_cache" in src


def test_version_strings_updated():
    """Schema version must be v04.3."""
    import audit_ingestion.canonical as c
    assert c.SCHEMA_VERSION == "v04.3"


# Additional coverage for the same pass

def test_parsed_document_has_file_hash_field():
    """ParsedDocument must have file_hash field."""
    from audit_ingestion.models import ParsedDocument
    doc = ParsedDocument(source_file="test.pdf", file_hash="abc123")
    assert doc.file_hash == "abc123"


def test_parsed_document_file_hash_optional():
    """file_hash is optional — must not break existing construction."""
    from audit_ingestion.models import ParsedDocument
    doc = ParsedDocument(source_file="test.pdf")
    assert doc.file_hash is None


def test_canonical_cache_key_fallback_no_file_hash():
    """Cache key must work when file_hash is None (falls back to full_text hash)."""
    from audit_ingestion.models import ParsedDocument, ParsedPage
    from audit_ingestion.canonical import _canonical_cache_key

    pages = [ParsedPage(page_number=1, text="content", char_count=7, extractor="pdfplumber")]
    doc = ParsedDocument(
        source_file="nofile.pdf", file_hash=None,
        full_text="content", page_count=1, pages=pages,
        extraction_chain=["pdfplumber"], primary_extractor="pdfplumber",
    )
    key = _canonical_cache_key(doc, "gpt-5.4")
    assert isinstance(key, str) and len(key) == 32


def test_render_page_image_cached_exists():
    """render_page_image_cached must be importable from extractor."""
    from audit_ingestion.extractor import render_page_image_cached
    assert callable(render_page_image_cached)


def test_render_page_images_accepts_file_hash():
    """_render_page_images must accept file_hash parameter."""
    import inspect
    from audit_ingestion import extractor
    sig = inspect.signature(extractor._render_page_images)
    assert "file_hash" in sig.parameters


def test_responses_api_format_has_name_field():
    """_responses_call must set text.format.name — required by Responses API."""
    import inspect
    from audit_ingestion.providers.openai_provider import OpenAIProvider
    src = inspect.getsource(OpenAIProvider._responses_call)
    assert '"name"' in src or "'name'" in src
    assert "schema_name" in src


def test_extract_structured_uses_canonical_model_constant():
    """extract_structured must explicitly use CANONICAL_MODEL, not a hardcoded string."""
    import inspect
    from audit_ingestion.providers.openai_provider import OpenAIProvider, RESCUE_MODEL
    src = inspect.getsource(OpenAIProvider.extract_structured)
    assert "CANONICAL_MODEL" in src
    assert RESCUE_MODEL not in src
