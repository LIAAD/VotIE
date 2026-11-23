"""
Configuration for Gemini span extraction via langextract.

Uses langextract's GeminiLanguageModel with a GEMINI_API_KEY.
"""

import os


class GeminiConfig:
    """Configuration for Gemini API extractor via langextract."""

    def __init__(
        self,
        api_key: str = None,
        # Must match the model actually reported in the paper. The previous
        # default (gemini-2.5-flash-preview-04-17) was never used in any reported
        # run — the CLI default overrode it — so a reproducer silently ran Flash.
        model_id: str = "gemini-2.5-pro",
        temperature: float = 0.0,
        max_char_buffer: int = 20000,
        timeout: int = 180,
        max_retries: int = 3,
        retry_delay: int = 5,
        max_output_tokens: int = None,
    ):
        """
        Initialize langextract Gemini configuration.

        Args:
            api_key: Gemini API key (falls back to GEMINI_API_KEY env var)
            model_id: Gemini model ID
            temperature: Sampling temperature (0.0 = deterministic)
            max_char_buffer: Max characters per chunk for langextract segmentation
            timeout: Per-request timeout in seconds. google-genai passes its
                     ``HttpOptions.timeout`` straight to ``httpx.Client.request``,
                     and when unset that is an explicit ``timeout=None`` — i.e. no
                     read timeout at all. A stalled response then blocks the whole
                     extraction loop indefinitely. 180s is ~5x the
                     ~37s a 2k-char document actually takes on gemini-2.5-pro.
            max_retries: Maximum attempts per document (including the first).
            retry_delay: Delay between attempts in seconds.
            max_output_tokens: Optional cap on tokens per chunk, as a circuit breaker
                               on runaway generation. Left off by default: on 2.5 Pro
                               the cap also covers thinking tokens (this dataset runs
                               ~5k thoughts for ~100 tokens of answer), so a cap tight
                               enough to catch a loop would truncate legitimate
                               documents and silently change the reported numbers.
                               The timeout bounds the wall clock instead.
        """
        self.api_key = api_key or os.environ.get("GEMINI_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_KEY environment variable, "
                "add it to .env, or pass api_key parameter."
            )
        self.model_id = model_id
        self.temperature = temperature
        self.max_char_buffer = max_char_buffer
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_output_tokens = max_output_tokens

    def __repr__(self) -> str:
        return (
            f"GeminiConfig(model={self.model_id}, "
            f"temperature={self.temperature}, max_char_buffer={self.max_char_buffer}, "
            f"timeout={self.timeout}s, max_retries={self.max_retries})"
        )
