"""
Routing decision engine.

The oracle IS the router. It decides whether it can handle a request
itself or whether to escalate to the heavy local backend (or cloud).

Pre-checks handle two hard gates:
  1. PII detected → never goes to cloud
  2. Input too long → straight to heavy local (oracle can't handle it well)

Everything else: the oracle model is asked "can you handle this?"
"""

import json
import re
from dataclasses import dataclass

import tiktoken
from loguru import logger

_enc = tiktoken.get_encoding("cl100k_base")

ROUTING_PROMPT = """You are deciding whether you can handle this request yourself or whether it needs a more powerful model.

You are a capable but lightweight model. Answer with ONLY a JSON object.

If you can handle it (simple questions, conversation, short tasks, factual lookups, basic explanations):
{"escalate": false, "reason": "<one sentence>"}

If it needs a more powerful model (code generation, complex reasoning, long-form writing, multi-step analysis, technical depth):
{"escalate": true, "reason": "<one sentence>"}"""


@dataclass
class RoutingDecision:
    escalate: bool
    reason: str
    pii_detected: bool = False
    estimated_tokens: int = 0


def _estimate_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += len(_enc.encode(content))
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += len(_enc.encode(part.get("text", "")))
    return total


def _extract_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                return " ".join(p.get("text", "") for p in content if p.get("type") == "text")
    return ""


def route_fast(messages: list[dict], pii_detected: bool = False,
               complexity_threshold: int = 800) -> RoutingDecision | None:
    """
    Hard gates only. Returns None if the oracle should decide.
    """
    tokens = _estimate_tokens(messages)

    # PII gate: keep it local, never cloud
    if pii_detected:
        return RoutingDecision(
            escalate=False,
            reason="PII detected — handling locally",
            pii_detected=True,
            estimated_tokens=tokens,
        )

    # Too long for the oracle to handle well → escalate
    if tokens > complexity_threshold:
        return RoutingDecision(
            escalate=True,
            reason=f"Input too long ({tokens} tokens) — escalating to heavy local",
            estimated_tokens=tokens,
        )

    return None


async def route_intelligent(messages: list[dict], client, model: str,
                            pii_detected: bool = False) -> RoutingDecision:
    """
    Ask the oracle: can you handle this, or should it go to the heavy backend?
    """
    tokens = _estimate_tokens(messages)
    user_text = _extract_user_text(messages)

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ROUTING_PROMPT},
                {"role": "user", "content": user_text[:2000]},
            ],
            max_tokens=300,
            temperature=0.0,
        )

        msg = resp.choices[0].message
        raw = (msg.content or "").strip()

        if not raw:
            # Handle thinking models (qwen3) that put everything in reasoning field
            reasoning = getattr(msg, "reasoning", None) or ""
            json_match = re.search(r'\{[^}]*"escalate"[^}]*\}', reasoning)
            if json_match:
                raw = json_match.group(0)
            else:
                lower = reasoning.lower()
                needs_escalation = any(w in lower for w in [
                    "need a more powerful", "too complex", "escalate",
                    "requires a more", "beyond my", "heavy model",
                    "code generation", "too long", "not feasible",
                ])
                raw = json.dumps({"escalate": needs_escalation, "reason": "Inferred from reasoning"})

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        parsed = json.loads(raw)
        return RoutingDecision(
            escalate=parsed.get("escalate", False),
            reason=parsed.get("reason", ""),
            pii_detected=pii_detected,
            estimated_tokens=tokens,
        )

    except Exception as e:
        logger.warning(f"Routing decision failed ({e}), handling locally")
        return RoutingDecision(
            escalate=False,
            reason="Routing failed — handling locally",
            pii_detected=pii_detected,
            estimated_tokens=tokens,
        )
