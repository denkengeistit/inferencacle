"""
Routing decision engine.

The oracle IS the router. It decides whether it can handle a request
itself or whether to escalate to the heavy local backend (or cloud).

Pre-checks handle two hard gates:
  1. PII detected → never goes to cloud
  2. Input too long → straight to heavy local (oracle can't handle it well)

Everything else: the oracle model is asked "can you handle this?"
"""
import os
import sys
import json
import logging
from pathlib import Path
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.schemas.messages import ChatMessage, RoutingDecision

log = logging.getLogger("oracle.router")

COMPLEXITY_THRESHOLD = int(os.getenv("COMPLEXITY_TOKEN_THRESHOLD", "800"))
PII_BLOCK_CLOUD      = os.getenv("PII_BLOCK_CLOUD", "true").lower() == "true"
CLOUD_ENABLED        = os.getenv("CLOUD_ENABLED", "false").lower() == "true"
ORACLE_MODEL         = os.getenv("ORACLE_MODEL", "qwen3-vl:8b")
LOCAL_HEAVY_MODEL    = os.getenv("LOCAL_HEAVY_MODEL", "zai-org/glm-4.7-flash")
CLOUD_MODEL          = os.getenv("CLOUD_MODEL", "anthropic/claude-sonnet-4-5")

_enc = tiktoken.get_encoding("cl100k_base")

ROUTING_PROMPT = """You are deciding whether you can handle this request yourself or whether it needs a more powerful model.

You are a capable but lightweight model. Answer with ONLY a JSON object.

If you can handle it (simple questions, conversation, short tasks, factual lookups, basic explanations):
{"escalate": false, "reason": "<one sentence>"}

If it needs a more powerful model (code generation, complex reasoning, long-form writing, multi-step analysis, technical depth):
{"escalate": true, "reason": "<one sentence>"}"""


def _estimate_tokens(messages: list[ChatMessage]) -> int:
    return sum(len(_enc.encode(m.content)) for m in messages)


def _extract_user_text(messages: list[ChatMessage]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


def route_fast(messages: list[ChatMessage], pii_detected: bool = False) -> RoutingDecision | None:
    """
    Hard gates only. Returns None if the oracle should decide.
    """
    tokens = _estimate_tokens(messages)

    # PII gate: keep it local, never cloud
    if pii_detected and PII_BLOCK_CLOUD:
        return RoutingDecision(
            target="self",
            model=ORACLE_MODEL,
            reason="PII detected — handling locally",
            pii_detected=True,
            estimated_tokens=tokens,
        )

    # Too long for the oracle to handle well → heavy local
    if tokens > COMPLEXITY_THRESHOLD:
        return RoutingDecision(
            target="local_heavy",
            model=LOCAL_HEAVY_MODEL,
            reason=f"Input too long ({tokens} tokens) — escalating to heavy local",
            estimated_tokens=tokens,
        )

    return None


async def route_intelligent(messages: list[ChatMessage], client, pii_detected: bool = False) -> RoutingDecision:
    """
    Ask the oracle: can you handle this, or should it go to the heavy backend?
    """
    tokens = _estimate_tokens(messages)
    user_text = _extract_user_text(messages)

    try:
        resp = await client.chat.completions.create(
            model=ORACLE_MODEL,
            messages=[
                {"role": "system", "content": ROUTING_PROMPT},
                {"role": "user", "content": user_text[:2000]},
            ],
            max_tokens=300,
            temperature=0.0,
        )

        msg = resp.choices[0].message
        # Some models (qwen3) put all output in a reasoning field and leave content empty
        raw = (msg.content or "").strip()
        if not raw:
            reasoning = getattr(msg, "reasoning", None) or ""
            # Try to find JSON in the reasoning text
            import re
            json_match = re.search(r'\{[^}]*"escalate"[^}]*\}', reasoning)
            if json_match:
                raw = json_match.group(0)
            else:
                # Infer from reasoning: if the model talks about escalating/needing more power
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
        escalate = parsed.get("escalate", False)
        reason = parsed.get("reason", "")

        if escalate:
            return RoutingDecision(
                target="local_heavy",
                model=LOCAL_HEAVY_MODEL,
                reason=reason,
                pii_detected=pii_detected,
                estimated_tokens=tokens,
            )
        else:
            return RoutingDecision(
                target="self",
                model=ORACLE_MODEL,
                reason=reason,
                pii_detected=pii_detected,
                estimated_tokens=tokens,
            )

    except Exception as e:
        log.warning(f"Routing decision failed ({e}), handling locally")
        return RoutingDecision(
            target="self",
            model=ORACLE_MODEL,
            reason="Routing failed — handling locally",
            pii_detected=pii_detected,
            estimated_tokens=tokens,
        )
