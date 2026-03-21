"""
Prompt coach: compresses and re-frames messages before sending upstream.
Reduces token count on cloud calls and injects model-specific best-practice framing.
"""
import sys
from pathlib import Path

# Add shared to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.schemas.messages import ChatMessage

COMPRESSION_SYSTEM = """You are a prompt optimizer.
Given a conversation, rewrite it as the most concise, clear request possible.
- Remove redundancy and filler
- Preserve all factual constraints and intent
- Return ONLY the rewritten user message. No explanation."""

async def compress(messages: list[ChatMessage], client) -> list[ChatMessage]:
    """
    Use the oracle itself to compress the outbound prompt.
    Only applied when routing to cloud (token saving matters most there).
    Skips compression for short messages (not worth the extra call).
    """
    user_messages = [m for m in messages if m.role == "user"]
    total_chars = sum(len(m.content) for m in user_messages)

    if total_chars < 500:
        return messages  # not worth compressing

    last_user = user_messages[-1].content
    resp = await client.chat.completions.create(
        model=None,  # uses oracle default
        messages=[
            {"role": "system", "content": COMPRESSION_SYSTEM},
            {"role": "user", "content": last_user},
        ],
        max_tokens=512,
        temperature=0.1,
    )
    compressed = resp.choices[0].message.content.strip()

    # Replace only the last user message; keep history for context
    result = []
    replaced = False
    for m in reversed(messages):
        if m.role == "user" and not replaced:
            result.insert(0, ChatMessage(role="user", content=compressed))
            replaced = True
        else:
            result.insert(0, m)
    return result
