"""
audit_ingestion_v03/audit_ingestion/providers/openai_provider.py
OpenAI provider — Responses API + Structured Outputs.

Model constants — change here only:
  CANONICAL_MODEL      = "gpt-5.4"       canonical structured extraction
  VISION_MODEL         = "gpt-5.4"       weak-page vision transcription
  RESCUE_MODEL         = "gpt-5.4-pro"   optional manual rescue (future)
  DEFAULT_MODEL        = CANONICAL_MODEL
"""
from __future__ import annotations
import base64
import json
import logging
import os
import subprocess
import tempfile
from typing import Optional

from .base import AIProvider

logger = logging.getLogger(__name__)

# ── Model constants ───────────────────────────────────────────────────────────
CANONICAL_MODEL = "gpt-5.4"
VISION_MODEL    = "gpt-5.4"
RESCUE_MODEL    = "gpt-5.4-pro"
DEFAULT_MODEL   = CANONICAL_MODEL

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class OpenAIProvider(AIProvider):

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        if not HAS_OPENAI:
            raise ImportError("openai not installed. Run: pip install openai")
        self.client = openai.OpenAI(api_key=api_key)
        self.model  = model
        logger.info(f"OpenAI provider ready — model: {model}")

    # ── Core Responses API call ───────────────────────────────────────────────

    def _responses_call(
        self,
        *,
        system: str,
        user: str,
        model: Optional[str] = None,
        max_output_tokens: int = 4000,
        json_schema: Optional[dict] = None,
    ) -> str:
        """
        Call OpenAI Responses API.
        Uses structured outputs when json_schema is provided.
        Returns raw response text.
        """
        m = model or self.model
        input_messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

        kwargs: dict = dict(
            model=m,
            input=input_messages,
            max_output_tokens=max_output_tokens,
        )

        if json_schema:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "json_schema": json_schema,
                }
            }

        resp = self.client.responses.create(**kwargs)
        # Responses API returns output_text directly
        return resp.output_text or ""

    # ── Structured canonical extraction ──────────────────────────────────────

    def extract_structured(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict,
        max_tokens: int = 4000,
    ) -> dict:
        """
        Extract structured JSON via Responses API + Structured Outputs.
        Returns validated dict. Raises on failure.
        """
        raw = self._responses_call(
            system=system,
            user=user,
            model=CANONICAL_MODEL,
            max_output_tokens=max_tokens,
            json_schema=json_schema,
        )
        if not raw or not raw.strip():
            raise ValueError("Empty response from structured extraction")
        return json.loads(raw)

    # ── Vision page transcription ─────────────────────────────────────────────

    def extract_text_from_page_images(
        self,
        *,
        images: list[bytes],
        prompt: str,
        model: Optional[str] = None,
    ) -> str:
        """
        Send page images to VISION_MODEL and extract text.
        Uses Responses API with image inputs.
        """
        if not images:
            return ""

        m = model or VISION_MODEL

        # Build image content blocks
        content: list[dict] = []
        for img_bytes in images[:8]:
            b64 = base64.b64encode(img_bytes).decode()
            content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{b64}",
                "detail": "high",
            })
        content.append({"type": "input_text", "text": prompt})

        resp = self.client.responses.create(
            model=m,
            input=[{"role": "user", "content": content}],
            max_output_tokens=4000,
        )
        return resp.output_text or ""

    def extract_text_from_pdf_vision(
        self,
        pdf_bytes: bytes,
        max_pages: int = 2,
    ) -> str:
        """Convert PDF pages to images and extract via vision. Used as last-resort fallback."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        img_dir = tempfile.mkdtemp()
        image_bytes_list: list[bytes] = []

        try:
            subprocess.run(
                ["pdftoppm", "-jpeg", "-r", "150", "-l", str(max_pages),
                 tmp_path, f"{img_dir}/page"],
                capture_output=True, timeout=60,
            )
            for img_file in sorted(os.listdir(img_dir)):
                if img_file.endswith(".jpg"):
                    img_path = os.path.join(img_dir, img_file)
                    with open(img_path, "rb") as f:
                        image_bytes_list.append(f.read())
                    os.unlink(img_path)
                    if len(image_bytes_list) >= max_pages:
                        break
        except Exception as e:
            logger.warning(f"pdftoppm failed: {e}")
        finally:
            try:
                os.unlink(tmp_path)
                os.rmdir(img_dir)
            except Exception:
                pass

        if not image_bytes_list:
            return ""

        return self.extract_text_from_page_images(
            images=image_bytes_list,
            prompt=(
                "Extract ALL text from these document pages faithfully. "
                "Preserve all numbers, dates, names, amounts, and terms exactly as written. "
                "Separate pages with '--- PAGE BREAK ---'."
            ),
        )
