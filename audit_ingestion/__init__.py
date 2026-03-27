from .router import ingest_one
from .models import AuditEvidence, IngestionResult, ParsedDocument
from .providers import get_provider

__version__ = "4.2.0"
__all__ = ["ingest_one", "AuditEvidence", "IngestionResult", "ParsedDocument", "get_provider"]
