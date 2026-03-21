from pydantic import BaseModel, Field
from typing import Optional, Literal, List

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class OracleRequest(BaseModel):
    """What the edge client sends to the oracle."""
    messages: List[ChatMessage]
    model: Optional[str] = None          # hint; oracle may override
    stream: bool = True
    max_tokens: int = 1024
    metadata: dict = Field(default_factory=dict)  # routing hints, session id, etc.

class RoutingDecision(BaseModel):
    target: Literal["self", "local_heavy", "cloud"]
    model: str
    reason: str
    pii_detected: bool = False
    estimated_tokens: int = 0

class RedactedPayload(BaseModel):
    messages: List[ChatMessage]
    redaction_map: dict = Field(default_factory=dict)  # placeholder → original
    pii_detected: bool = False
