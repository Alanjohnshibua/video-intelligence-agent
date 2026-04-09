"""
sarvam_client.py
----------------
Reusable Sarvam AI API wrapper built on the ``sarvamai`` SDK.

Features
--------
* API key loaded from the ``SARVAM_API_KEY`` environment variable or passed directly
* Automatic retries with exponential back-off
* Structured CCTV prompt builder parsing into standard OpenAI-style roles
* Implements ``ReasoningResponderProtocol`` (defined in ``reasoning.py``)
* Graceful ``SarvamClientError`` on unrecoverable failure
"""

from __future__ import annotations

import os
import time as _time
from typing import Any

from video_intelligence_agent.cctv_pipeline.utils.logger import get_pipeline_logger

logger = get_pipeline_logger("agent.sarvam_client")


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class SarvamClientError(RuntimeError):
    """Raised when the Sarvam API call fails after all retries."""

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} (caused by: {self.cause})" if self.cause else base


# ---------------------------------------------------------------------------
# Main client class
# ---------------------------------------------------------------------------

class SarvamClient:
    """
    Thin, robust wrapper around the ``sarvamai`` SDK.

    Parameters
    ----------
    api_key:
        Sarvam API subscription key. When omitted the ``SARVAM_API_KEY``
        environment variable is used.
    model_name:
        Sarvam model identifier (default: ``sarvam-30b`` or ``sarvam-105b``).
    max_retries:
        Number of retry attempts on transient API errors.
    retry_delay_seconds:
        Base back-off delay in seconds (doubled on each retry).
    temperature:
        Sampling temperature for the model (0.0 = deterministic).
    max_output_tokens:
        Upper bound on the number of tokens generated per response.
    """

    DEFAULT_MODEL = "sarvam-105b"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model_name: str = DEFAULT_MODEL,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
        temperature: float = 0.2,
        max_output_tokens: int = 4096,
    ) -> None:
        self._api_key = api_key or os.environ.get("SARVAM_API_KEY", "")
        if not self._api_key:
            raise SarvamClientError(
                "Sarvam API key is missing. Set the SARVAM_API_KEY environment "
                "variable or pass api_key= at construction time."
            )

        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        self._client: Any = None   # lazily initialised
        self._initialised = False

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_initialised(self) -> None:
        """Lazy-initialise the sarvamai client."""
        if self._initialised:
            return

        try:
            from sarvamai import SarvamAI
        except ImportError as exc:
            raise SarvamClientError(
                "The 'sarvamai' package is not installed. "
                "Install it with: pip install sarvamai",
                cause=exc,
            ) from exc

        try:
            self._client = SarvamAI(api_subscription_key=self._api_key)
            self._initialised = True
            logger.info("Sarvam AI client initialised | model=%s", self.model_name)
        except Exception as exc:
            raise SarvamClientError(
                "Failed to initialise the Sarvam client.",
                cause=exc,
            ) from exc

    # ------------------------------------------------------------------
    # ReasoningResponderProtocol implementation
    # ------------------------------------------------------------------

    def generate(self, *, prompt: str | list[dict[str, str]]) -> str:
        """
        Send *prompt* to Sarvam and return the response text.

        Parameters
        ----------
        prompt:
            Either a simple string, or a list of dictionaries representing
            the conversation history (e.g. `[{"role": "user", "content": ...}]`).
        """
        self._ensure_initialised()

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        delay = self.retry_delay_seconds
        last_exc: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "Sarvam API call | attempt=%d/%d model=%s",
                    attempt,
                    self.max_retries,
                    self.model_name,
                )
                response = self._client.chat.completions(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                )
                text = _extract_text(response)
                logger.info("Sarvam API call succeeded | response_chars=%d", len(text))
                return text

            except SarvamClientError:
                raise  # don't retry extraction errors
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Sarvam API error (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    _time.sleep(delay)
                    delay *= 2  # exponential back-off

        raise SarvamClientError(
            f"Sarvam API failed after {self.max_retries} attempts.",
            cause=last_exc,
        )

    # ------------------------------------------------------------------
    # Prompt construction helpers (used by AgentController)
    # ------------------------------------------------------------------

    @staticmethod
    def build_cctv_prompt(
        query: str,
        events_json: str,
        *,
        extra_context: str = "",
    ) -> list[dict[str, str]]:
        """
        Format a structured CCTV reasoning prompt into standard role messages.
        """
        system_prompt = (
            "You are an expert CCTV security analyst assistant. "
            "Your primary goal is to provide high-quality narrative summaries of events. "
            "Analyze the provided JSON evidence and explain clearly what happened in natural language.\n\n"
            "GUIDELINES:\n"
            "1. Answer strictly based on the provided JSON data.\n"
            "2. Do NOT invent events, names, or locations.\n"
            "3. Prioritize a clear, human-readable summary of actions and behaviors.\n"
            "4. Include timestamps and durations naturally in your explanation.\n"
            "5. Mention clip paths only at the end or if they are essential to the description.\n"
            "6. Be honest if the evidence is insufficient.\n\n"
            "CRITICAL: Always adhere STRICTLY to the user's requested format. "
            "If they ask for a single sentence, do NOT provide a list."
        )

        user_content = [
            "CCTV EVENT DATA (JSON):",
            "```json",
            events_json,
            "```",
        ]

        if extra_context.strip():
            user_content.extend([
                "\nADDITIONAL CONTEXT:",
                extra_context.strip(),
            ])

        user_content.extend([
            f"\nUSER QUESTION: {query}"
        ])

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(user_content)},
        ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_text(response: Any) -> str:
    """Safely extract text from Sarvam response by converting to dict first."""
    import json
    
    # Safely convert to dictionary to bypass Pydantic hiding non-standard fields
    raw_dict = {}
    if isinstance(response, dict):
        raw_dict = response
    elif hasattr(response, "model_dump"):
        raw_dict = response.model_dump(exclude_none=False)
    elif hasattr(response, "dict"):
        raw_dict = response.dict(exclude_none=False)
    elif hasattr(response, "to_dict"):
        raw_dict = response.to_dict()
    else:
        raw_dict = getattr(response, "__dict__", {})

    content_parts = []
    
    try:
        choices = raw_dict.get("choices", [])
        if choices:
            choice = choices[0]
            msg = choice.get("message", {}) or {}
            
            # Extract standard content
            s_content = msg.get("content")
            if s_content:
                content_parts.append(str(s_content).strip())
            else:
                # Failsafe: Only show reasoning if the model ran out of tokens before writing the actual answer
                r_content = msg.get("reasoning_content")
                if r_content:
                    content_parts.append(f"⚠️ Model ran out of tokens before finishing. Last thoughts:\n{str(r_content)[-1000:]}")
                
            if content_parts:
                return "\n\n".join(content_parts).strip()
    except Exception:
        pass

    try:
        raw_debug = json.dumps(raw_dict, default=str)
    except Exception:
        raw_debug = repr(response)

    raise SarvamClientError(f"Sarvam returned an empty or unreadable response.\n\nStructure: {raw_debug[:1500]}")
