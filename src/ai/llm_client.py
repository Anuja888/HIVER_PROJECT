"""
Phase 5 — LLM Client Wrapper.

Thin wrapper supporting:
  - mock mode (deterministic canned responses, zero cost, works without API key)
  - OpenAI (gpt-4o-mini default)
  - Anthropic (claude-3-5-sonnet default)

Features:
  - Automatic retries with exponential backoff on rate-limit / transient errors
  - Configurable timeout
  - Structured JSON output with pydantic validation + graceful repair/retry
  - All calls return a standard ChatResponse object

Decision log:
  - Mock mode uses a hash of the prompt to pick deterministic canned responses,
    so repeated calls with the same prompt always return the same result. This
    makes the eval harness reproducible without spending API credits.
  - Retry logic: up to 3 attempts, 2^attempt * 1s backoff. Rate limit (429) and
    server errors (500, 502, 503) trigger a retry; auth errors (401) do not.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

from pydantic import BaseModel

from src.config.settings import settings

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------

class ChatResponse(BaseModel):
    content: str
    model: str
    provider: str
    mock: bool = False
    tokens_used: int = 0


# ---------------------------------------------------------------------------
# Mock responses — deterministic, no API required
# ---------------------------------------------------------------------------

_MOCK_INTENT_RESPONSES: list[dict] = [
    {"intent": "order_delivery_tracking", "confidence": 0.85,
     "reason": "Customer asks about package delivery status.", "needs_context": False},
    {"intent": "product_return_refund", "confidence": 0.80,
     "reason": "Customer reports receiving wrong item and wants refund.", "needs_context": False},
    {"intent": "technical_bug_malfunction", "confidence": 0.75,
     "reason": "Customer reports app crash.", "needs_context": False},
    {"intent": "account_access_login", "confidence": 0.82,
     "reason": "Customer cannot log in to account.", "needs_context": False},
    {"intent": "prime_subscription_billing", "confidence": 0.78,
     "reason": "Customer asking about Prime cancellation charge.", "needs_context": False},
    {"intent": "product_question_usage", "confidence": 0.70,
     "reason": "Customer asking about product compatibility.", "needs_context": False},
    {"intent": "general_feedback_other", "confidence": 0.55,
     "reason": "Message doesn't map clearly to a specific intent.", "needs_context": True},
    {"intent": "account_security_compromise", "confidence": 0.88,
     "reason": "Customer reports unauthorized account activity.", "needs_context": False},
    {"intent": "seller_third_party_issue", "confidence": 0.73,
     "reason": "Customer complains about third-party seller.", "needs_context": False},
]

_MOCK_REPLY_TEMPLATE = (
    "Thank you for reaching out to Amazon customer support. Based on similar "
    "cases, I can see that [RESOLUTION_PATTERN]. Please [ACTION_STEP]. "
    "If this doesn't resolve your issue, we're happy to escalate. "
    "Is there anything else I can help you with today?"
)

_MOCK_JUDGE_RESPONSE: dict = {
    "relevance": 4, "correctness": 3, "grounding": 4,
    "helpfulness": 3, "completeness": 3, "tone": 5,
    "no_unsupported_claims": 4, "overall": 3.7,
    "reasoning": "The reply addresses the customer's concern and is grounded in retrieved evidence. Minor deduction for vague action step."
}


def _mock_hash(prompt: str) -> int:
    return int(hashlib.md5(prompt.encode()).hexdigest(), 16)


def _mock_response(prompt: str) -> str:
    """Deterministic mock response based on prompt content."""
    p_lower = prompt.lower()
    if "intent" in p_lower and "classify" in p_lower:
        idx = _mock_hash(prompt) % len(_MOCK_INTENT_RESPONSES)
        return json.dumps(_MOCK_INTENT_RESPONSES[idx])
    if "judge" in p_lower or "rubric" in p_lower or "score" in p_lower:
        return json.dumps(_MOCK_JUDGE_RESPONSE)
    if "reply" in p_lower or "respond" in p_lower or "resolution" in p_lower:
        return _MOCK_REPLY_TEMPLATE
    if "grounding" in p_lower or "hallucin" in p_lower:
        return json.dumps({"grounded": True, "flags": [], "score": 0.82})
    if "escalat" in p_lower or "route" in p_lower:
        return json.dumps({
            "decision": "AUTO_HANDLE", "reason": "High confidence intent, good retrieval match.",
            "risk_flags": [], "confidence": 0.78
        })
    # generic
    return "I understand your concern. Let me help you with that. [MOCK RESPONSE]"


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

def _call_openai(messages: list[dict], model: str, temperature: float,
                 max_tokens: int, timeout: float) -> ChatResponse:
    try:
        import openai
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    client = openai.OpenAI(
        api_key=settings.openai_api_key,
        timeout=timeout,
    )
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            return ChatResponse(
                content=content,
                model=model,
                provider="openai",
                tokens_used=resp.usage.total_tokens if resp.usage else 0,
            )
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status in (429, 500, 502, 503) and attempt < 2:
                wait = 2 ** attempt
                log.warning("OpenAI error %s — retry in %ss", status, wait)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("OpenAI call failed after 3 attempts")


# ---------------------------------------------------------------------------
# Anthropic client
# ---------------------------------------------------------------------------

def _call_anthropic(messages: list[dict], model: str, temperature: float,
                    max_tokens: int, timeout: float) -> ChatResponse:
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    # Anthropic uses a system message separately
    system_msgs = [m["content"] for m in messages if m["role"] == "system"]
    user_msgs = [m for m in messages if m["role"] != "system"]
    system_prompt = "\n\n".join(system_msgs) if system_msgs else None

    for attempt in range(3):
        try:
            kwargs: dict[str, Any] = dict(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=user_msgs,
            )
            if system_prompt:
                kwargs["system"] = system_prompt
            resp = client.messages.create(**kwargs)
            content = resp.content[0].text if resp.content else ""
            return ChatResponse(
                content=content,
                model=model,
                provider="anthropic",
                tokens_used=(resp.usage.input_tokens + resp.usage.output_tokens
                             if resp.usage else 0),
            )
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status in (429, 500, 502, 503) and attempt < 2:
                wait = 2 ** attempt
                log.warning("Anthropic error %s — retry in %ss", status, wait)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Anthropic call failed after 3 attempts")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chat(
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: float = 30.0,
    provider: str | None = None,
) -> ChatResponse:
    """
    Send a chat request. Uses provider from settings.llm_provider unless
    overridden. Falls back to mock if provider=='mock' or no API key set.

    Args:
        messages: List of {"role": "system"|"user"|"assistant", "content": str}
        model: Override model name (otherwise uses settings)
        temperature: Sampling temperature
        max_tokens: Max output tokens
        timeout: Request timeout in seconds
        provider: Override provider ('mock', 'openai', 'anthropic')
    """
    prov = provider or settings.llm_provider

    # Auto-fall-back to mock if no key
    if prov == "openai" and not settings.openai_api_key:
        log.warning("No OPENAI_API_KEY — falling back to mock mode")
        prov = "mock"
    if prov == "anthropic" and not settings.anthropic_api_key:
        log.warning("No ANTHROPIC_API_KEY — falling back to mock mode")
        prov = "mock"

    if prov == "mock":
        # Combine all messages into a single prompt string for mock routing
        full_prompt = " ".join(m.get("content", "") for m in messages)
        return ChatResponse(
            content=_mock_response(full_prompt),
            model="mock",
            provider="mock",
            mock=True,
        )

    if prov == "openai":
        m = model or settings.openai_model
        return _call_openai(messages, m, temperature, max_tokens, timeout)

    if prov == "anthropic":
        m = model or settings.anthropic_model
        return _call_anthropic(messages, m, temperature, max_tokens, timeout)

    raise ValueError(f"Unknown LLM provider: {prov!r}. Use 'mock', 'openai', or 'anthropic'.")


def chat_json(
    messages: list[dict],
    schema: type,
    **kwargs: Any,
) -> Any:
    """
    Like chat() but parses response as JSON and validates against a pydantic schema.
    Retries once with an explicit repair prompt if the first attempt is invalid JSON.

    Args:
        messages: Chat messages
        schema: Pydantic BaseModel class to validate against
        **kwargs: Passed to chat()

    Returns:
        Validated pydantic model instance
    """
    for attempt in range(3):
        resp = chat(messages, **kwargs)
        raw = resp.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = "\n".join(
                line for line in raw.splitlines()
                if not line.strip().startswith("```")
            ).strip()

        try:
            data = json.loads(raw)
            return schema(**data)
        except (json.JSONDecodeError, Exception) as e:
            if attempt < 2:
                log.warning("JSON parse failed (attempt %d): %s. Retrying with repair prompt.", attempt, e)
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content":
                     f"Your previous response was not valid JSON. "
                     f"Please respond with ONLY valid JSON matching this schema: "
                     f"{schema.model_json_schema()}. No other text."},
                ]
            else:
                log.error("JSON parse failed after 3 attempts: %s", e)
                raise ValueError(f"LLM returned invalid JSON after 3 attempts: {raw[:200]}")
    raise RuntimeError("Unreachable")
