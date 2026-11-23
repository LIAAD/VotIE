"""Configuration for GPT-5.5 span extraction via langextract."""

import os


def _from_env(name: str, fallback: str) -> str:
    """Environment override, ignoring a variable that is set but empty.

    `IAEDU_ENDPOINT=` left dangling in .env would otherwise resolve to "" and
    surface as an opaque requests error rather than falling back to the default.
    """
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else fallback


class GptConfig:
    """Configuration for GPT-5.5 via the IAEDU streaming API.

    All three gateway parameters are overridable from the environment (.env is
    loaded by the extraction scripts before this class is instantiated), so
    pointing a run at a different agent or channel needs no code change:

        IAEDU_API_KEY     value of the x-api-key header
        IAEDU_ENDPOINT    full stream URL, including the agent id
        IAEDU_CHANNEL_ID  channel the request is posted to

    Precedence is explicit argument > environment. Note
    this is the opposite of AmaliaConfig, where AMALIA_BASE_URL wins over an
    explicitly passed base_url.
    """

    # The gateway endpoint and channel are deployment-specific: set IAEDU_ENDPOINT
    # and IAEDU_CHANNEL_ID (or IAEDU_CREDENTIALS) in .env.
    ENDPOINT = None
    CHANNEL_ID = None
    # A label only. The agent decides the model; see the note in __init__.
    MODEL_ID = "gpt-5.5"

    def __init__(
        self,
        api_key: str = None,
        endpoint: str = None,
        channel_id: str = None,
        model_id: str = MODEL_ID,
        max_char_buffer: int = 20000,
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: int = 5,
        temperature: float = None,
        rate_limit_delay: int = 30,
        rate_limit_retries: int = 6,
        request_interval: float = 0.0,
    ):
        """
        Note on `model_id` and `temperature`:

        This endpoint is an agent gateway, not the OpenAI API. It accepts only
        `channel_id`, `thread_id`, `user_info` and `message`; the model is fixed
        by the configured agent. So `model_id` is a label recorded in the output,
        NOT a routing instruction — it cannot be used to select a model, and it is
        not evidence of which model served the request.

        `temperature` is therefore off by default: whether the gateway honours or
        rejects an extra field is untested. Probe with a 2-document run before
        enabling it, and if the gateway ignores it, say so in the paper rather
        than claiming greedy decoding.
        """
        self.channel_id = channel_id or _from_env("IAEDU_CHANNEL_ID", self.CHANNEL_ID)
        # Credential pool, rotated on rate limits (see GptLanguageModel).
        # IAEDU_CREDENTIALS=key1:channel1,key2:channel2  — all on the same agent/endpoint.
        # An explicit api_key always means a pool of one.
        pool = _from_env("IAEDU_CREDENTIALS", "")
        if api_key:
            self.credentials = [(api_key, self.channel_id)]
        elif pool:
            self.credentials = [tuple(p.strip().split(":", 1)) for p in pool.split(",") if p.strip()]
            if any(len(c) != 2 or not all(c) for c in self.credentials):
                raise ValueError("IAEDU_CREDENTIALS must be 'key:channel,key:channel,...'")
        else:
            key = _from_env("IAEDU_API_KEY", "")
            self.credentials = [(key, self.channel_id)] if key else []
        if not self.credentials:
            raise ValueError(
                "GPT API key required. Set IAEDU_API_KEY or IAEDU_CREDENTIALS in .env "
                "(or the shell) or pass api_key."
            )
        self.api_key, self.channel_id = self.credentials[0]
        self.endpoint = endpoint or _from_env("IAEDU_ENDPOINT", self.ENDPOINT)
        if not self.endpoint:
            raise ValueError("Gateway endpoint required. Set IAEDU_ENDPOINT in .env (or the shell) "
                             "or pass endpoint.")
        self.model_id = model_id
        self.max_char_buffer = max_char_buffer
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        # The gateway returns 'Rate limit reached (429)' as an HTTP-200 error event.
        # Back off exponentially from rate_limit_delay; request_interval paces
        # requests client-side to stay under the limit in the first place.
        self.rate_limit_delay = rate_limit_delay
        self.rate_limit_retries = rate_limit_retries
        self.request_interval = request_interval

    def __repr__(self) -> str:
        # endpoint and channel_id are echoed because they are now environment
        # dependent: the run log is the only record of which gateway served it.
        return (
            f"GptConfig(model={self.model_id}, endpoint={self.endpoint}, "
            f"channel_id={self.channel_id}, credentials={len(self.credentials)}, "
            f"max_char_buffer={self.max_char_buffer}, "
            f"temperature={self.temperature if self.temperature is not None else 'gateway default'})"
        )
