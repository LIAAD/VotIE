"""
Gemini-based span extractor using langextract with GeminiLanguageModel.

Mirrors the AmaliaSpanExtractor flow but wired to Gemini via langextract,
enabling a fair comparison of langextract outputs across providers.
"""

import math
import threading
import time
import logging
from typing import Optional

import langextract as lx
from google.genai import types as genai_types
from langextract.providers.gemini import GeminiLanguageModel
from langextract.prompt_validation import PromptValidationLevel

from ..shared.base_extractor import BaseSpanExtractor
from ..schemas import SpanEntity, ENTITY_TYPES
from ..span_alignment import align_extraction_result
from ..amalia.prompts import get_prompt_description
from ..fixed_examples import get_fixed_examples
from ..usage import attach_gemini_usage
from .config import GeminiConfig


logger = logging.getLogger(__name__)


class DocumentTimeout(Exception):
    """Raised when a document exceeds its wall-clock budget and is abandoned."""


class GeminiSpanExtractor(BaseSpanExtractor):
    """Gemini-based span extractor via langextract."""

    def __init__(
        self,
        config: Optional[GeminiConfig] = None,
        strategy: str = "zero_shot",
        num_examples: int = 5,
    ):
        """
        Args:
            config: Gemini configuration
            strategy: "zero_shot" or "few_shot"
            num_examples: Number of fixed few-shot examples to use (max 5)
        """
        self.config = config or GeminiConfig()

        super().__init__(model_id=self.config.model_id, strategy=strategy)

        self.prompt_description = get_prompt_description()
        # langextract 1.1.1 requires at least one example unconditionally.
        # For zero_shot we use 1 example as the minimum required scaffold.
        self.lx_examples = get_fixed_examples(num_examples if strategy == "few_shot" else 1)

        # response_mime_type forces JSON output at the API level; without it,
        # GeminiLanguageModel only sets this when a gemini_schema is attached,
        # which can silently produce empty responses when schema constraints fail.
        #
        # http_options is what stops a stalled generation from blocking forever:
        # google-genai forwards HttpOptions.timeout (ms) to httpx as the per-request
        # timeout, and leaves it at None otherwise. Retries stay off here so they
        # happen in _call_with_retry, where they are logged and counted.
        self._lm = GeminiLanguageModel(
            model_id=self.config.model_id,
            api_key=self.config.api_key,
            temperature=self.config.temperature,
            response_mime_type="application/json",
            http_options=genai_types.HttpOptions(
                timeout=self.config.timeout * 1000,
            ),
        )

        # max_output_tokens is not in langextract's Gemini kwargs allowlist, so it
        # has to go into _extra_kwargs directly; infer() reads it back from there.
        if self.config.max_output_tokens:
            self._lm._extra_kwargs["max_output_tokens"] = self.config.max_output_tokens

        # langextract discards the provider response, so token usage has to be
        # captured at the client. Reset per document, snapshot into diagnostics.
        self._usage = attach_gemini_usage(self._lm)

        logger.info(
            f"Initialized GeminiSpanExtractor with {self.config}, "
            f"strategy={strategy}, examples={len(self.lx_examples)}"
        )

    def _build_prompt(self, text: str, examples=None) -> str:
        """Not used directly — langextract handles prompt construction."""
        return self.prompt_description

    def _document_budget(self, text: str) -> float:
        """Wall-clock budget for one attempt at one document, in seconds.

        langextract splits a document into ceil(len/max_char_buffer) chunks and
        calls the API once per chunk, so the budget scales with chunk count. The
        slack covers prompt building and the between-chunk bookkeeping.
        """
        chunks = max(1, math.ceil(len(text) / self.config.max_char_buffer))
        return self.config.timeout * chunks + 30

    def _run_with_deadline(self, fn, deadline: float, document_id: str):
        """Run ``fn`` on a daemon thread and abandon it if it overruns.

        The httpx timeout covers a socket that goes quiet, but not every way a
        call can fail to return — a response that trickles in resets the read
        timeout on each chunk of bytes, and a wedged connection pool never issues
        a request to time out in the first place. This is the backstop that keeps
        one bad document from stopping the run: the thread is left to die on its
        own (it cannot be killed) while the loop moves on. It is a daemon so it
        can never hold up interpreter exit, and the usage tracker's generation
        counter discards its tokens if it does eventually return.
        """
        box = {}

        def target():
            try:
                box["result"] = fn()
            except BaseException as e:  # noqa: BLE001 - relayed to the caller
                box["error"] = e

        thread = threading.Thread(
            target=target, daemon=True, name=f"lx-extract-{document_id}"
        )
        thread.start()
        thread.join(deadline)

        if thread.is_alive():
            raise DocumentTimeout(
                f"no response after {deadline:.0f}s (abandoned)"
            )
        if "error" in box:
            raise box["error"]
        return box["result"]

    def _call_with_retry(self, text: str, document_id: str):
        """Call langextract with a per-attempt deadline and bounded retries.

        Gemini stalls are transient — the same document usually succeeds on a
        fresh request — so unlike the AMALIA extractor we do retry timeouts.
        Worst case per document is max_retries x (budget + retry_delay).
        """
        deadline = self._document_budget(text)
        last_exc = None

        for attempt in range(1, self.config.max_retries + 1):
            attempt_start = time.time()
            try:
                return self._run_with_deadline(
                    lambda: lx.extract(
                        text_or_documents=text,
                        prompt_description=self.prompt_description,
                        examples=self.lx_examples,
                        model=self._lm,
                        max_char_buffer=self.config.max_char_buffer,
                        fence_output=False,
                        use_schema_constraints=False,
                        show_progress=False,
                        max_workers=1,
                        extraction_passes=1,
                        prompt_validation_level=PromptValidationLevel.OFF,
                    ),
                    deadline=deadline,
                    document_id=document_id,
                )
            except KeyboardInterrupt:
                raise
            except Exception as e:
                elapsed = time.time() - attempt_start
                msg = str(e).lower()
                transient = (
                    isinstance(e, DocumentTimeout)
                    or "timeout" in msg
                    or "timed out" in msg
                    or "connection" in msg
                    or "429" in msg
                    or "resource_exhausted" in msg
                    or "500" in msg
                    or "502" in msg
                    or "503" in msg
                    or "504" in msg
                    or "unavailable" in msg
                )
                if transient and attempt < self.config.max_retries:
                    logger.warning(
                        f"  {document_id}: attempt {attempt}/{self.config.max_retries} "
                        f"failed after {elapsed:.0f}s ({e}); retrying in "
                        f"{self.config.retry_delay}s"
                    )
                    last_exc = e
                    time.sleep(self.config.retry_delay)
                    continue
                raise

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Gemini call failed for {document_id} after retries")

    def extract(
        self,
        text: str,
        document_id: str,
        align_spans: bool = True,
    ):
        """
        Extract spans from text using Gemini via langextract.

        Args:
            text: Source text to extract from
            document_id: Document identifier
            align_spans: Whether to align spans to source text

        Returns:
            SpanExtractionResult with extracted entities
        """
        start_time = time.time()
        self._usage.reset()

        try:
            logger.info(f"Extracting from {document_id} ({len(text)} chars, strategy={self.strategy})")

            annotated_doc = self._call_with_retry(text, document_id)

            processing_time = time.time() - start_time
            result = self._convert_result(
                annotated_doc, document_id, text, processing_time
            )

            n_raw = len(annotated_doc.extractions) if annotated_doc.extractions else 0
            logger.info(
                f"  langextract returned {n_raw} extractions, "
                f"{len(result.entities)} valid entities ({processing_time:.1f}s)"
            )

            # Snapshot pre-alignment output so the span-discard rate is recoverable
            # from the saved predictions rather than only from stdout logs.
            usage = self._usage.snapshot()
            diagnostics = {
                "n_raw_generated": n_raw,
                "n_after_type_filter": len(result.entities),
                "n_dropped_unknown_type": n_raw - len(result.entities),
                "pre_alignment_entities": [
                    {"text": e.text, "type": e.type} for e in result.entities
                ],
                "raw_generated": [
                    {
                        "text": x.extraction_text,
                        "type": x.extraction_class,
                    }
                    for x in (annotated_doc.extractions or [])
                ],
                "usage": usage,
            }
            # Generation time alone; result.processing_time also covers alignment.
            result.api_time = usage["api_seconds"]

            if align_spans:
                result, alignment_stats = align_extraction_result(
                    result, strict=True
                )
                logger.info(
                    f"  Aligned {alignment_stats['aligned_spans']}/{alignment_stats['total_spans']} spans "
                    f"(fidelity={alignment_stats['alignment_rate']:.0%})"
                )
                diagnostics.update(
                    {
                        "n_before_alignment": alignment_stats["total_spans"],
                        "n_aligned": alignment_stats["aligned_spans"],
                        "n_dropped_alignment": alignment_stats["failed_spans"],
                        "alignment_rate": alignment_stats["alignment_rate"],
                        "failed_alignments": alignment_stats["failed_alignments"],
                    }
                )

            result.diagnostics = diagnostics
            return result

        except Exception as e:
            timed_out = isinstance(e, DocumentTimeout)
            processing_time = time.time() - start_time
            if timed_out:
                logger.error(
                    f"Skipping {document_id}: timed out after {processing_time:.0f}s "
                    f"over {self.config.max_retries} attempt(s) — {e}"
                )
            else:
                logger.error(f"Error extracting from {document_id}: {e}")
            result = self._create_result(
                document_id=document_id,
                text=text,
                entities=[],
                processing_time=processing_time,
                error=f"timeout: {e}" if timed_out else str(e),
            )
            # A failed document still burns tokens; record them so cost is not
            # under-reported and the failure rate is auditable per document.
            result.diagnostics = {
                "document_failed": True,
                "timed_out": timed_out,
                "usage": self._usage.snapshot(),
            }
            return result

    def _convert_result(self, annotated_doc, document_id, text, processing_time):
        """Convert langextract AnnotatedDocument to SpanExtractionResult."""
        entities = []
        if annotated_doc.extractions:
            for extraction in annotated_doc.extractions:
                entity_type = extraction.extraction_class
                entity_text = extraction.extraction_text
                if entity_type in ENTITY_TYPES and entity_text:
                    entities.append(
                        SpanEntity(
                            text=entity_text,
                            type=entity_type,
                            start=None,
                            end=None,
                        )
                    )

        logger.info(f"  Converted {len(entities)} entities from langextract result")

        return self._create_result(
            document_id=document_id,
            text=text,
            entities=entities,
            processing_time=processing_time,
        )
