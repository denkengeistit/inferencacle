"""
Routing decision engine.
Decides whether to handle a request locally (oracle self), 
route to the heavy local backend, or escalate to cloud.
"""
import os
import sys
from pathlib import Path
import tiktoken

# Add shared to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.schemas.messages import ChatMessage, RoutingDecision

COMPLEXITY_THRESHOLD = int(os.getenv("COMPLEXITY_TOKEN_THRESHOLD", "800"))
PII_BLOCK_CLOUD      = os.getenv("PII_BLOCK_CLOUD", "true").lower() == "true"
CLOUD_ENABLED        = os.getenv("CLOUD_ENABLED", "false").lower() == "true"
ORACLE_MODEL         = os.getenv("ORACLE_MODEL", "qwen2.5:32b")
LOCAL_HEAVY_MODEL    = os.getenv("LOCAL_HEAVY_MODEL", "qwen2.5:72b")
CLOUD_MODEL          = os.getenv("CLOUD_MODEL", "anthropic/claude-sonnet-4-5")

_enc = tiktoken.get_encoding("cl100k_base")

def _estimate_tokens(messages: list[ChatMessage]) -> int:
    return sum(len(_enc.encode(m.content)) for m in messages)

def route(messages: list[ChatMessage], pii_detected: bool = False) -> RoutingDecision:
    """
    Routing rules (ordered by priority):
    1. PII detected + cloud blocked → force local or self
    2. Cloud disabled → choose between self and heavy local
    3. Token count > threshold → heavy local
    4. Single-turn, short, factual → handle self
    5. Default → cloud (if enabled) or heavy local
    """
    tokens = _estimate_tokens(messages)

    # Rule 1: PII gate
    if pii_detected and PII_BLOCK_CLOUD:
        target = "local_heavy" if tokens > 300 else "self"
        return RoutingDecision(
            target=target,
            model=LOCAL_HEAVY_MODEL if target == "local_heavy" else ORACLE_MODEL,
            reason="PII detected — cloud routing blocked",
            pii_detected=True,
            estimated_tokens=tokens,
        )

    # Rule 2: Complexity gate
    if tokens > COMPLEXITY_THRESHOLD:
        return RoutingDecision(
            target="local_heavy",
            model=LOCAL_HEAVY_MODEL,
            reason=f"Token estimate {tokens} exceeds threshold {COMPLEXITY_THRESHOLD}",
            estimated_tokens=tokens,
        )

    # Rule 3: Short/simple — oracle handles self
    if tokens < 200 and len(messages) <= 2:
        return RoutingDecision(
            target="self",
            model=ORACLE_MODEL,
            reason="Short single-turn — oracle handles locally",
            estimated_tokens=tokens,
        )

    # Rule 4: Default routing
    if CLOUD_ENABLED:
        return RoutingDecision(
            target="cloud",
            model=CLOUD_MODEL,
            reason="Default routing to cloud",
            estimated_tokens=tokens,
        )
    else:
        return RoutingDecision(
            target="local_heavy",
            model=LOCAL_HEAVY_MODEL,
            reason="Default routing to heavy local (cloud disabled)",
            estimated_tokens=tokens,
        )
