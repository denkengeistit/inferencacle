"""
Oracle — FastAPI OpenAI-compatible chat completions proxy.
Implements: redaction → routing → compression → upstream call → stream back.
"""
import os
import sys
import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Make shared schemas importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI

from shared.schemas.messages import OracleRequest, ChatMessage
from src.redactor.redactor import redact
from src.router.router import route_fast, route_intelligent
from src.prompt_coach.coach import compress

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("oracle")

# ── Oracle system prompt ─────────────────────────────────────────────────────
_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parents[1] / "config" / "system_prompt.txt"
ORACLE_SYSTEM_PROMPT = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip() if _SYSTEM_PROMPT_PATH.exists() else ""

# ── Client pool ──────────────────────────────────────────────────────────────
def _make_client(base_url: str, api_key: str) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=base_url, api_key=api_key)

ORACLE_CLIENT = _make_client(
    os.getenv("ORACLE_API_BASE", "http://localhost:11434/v1"),
    os.getenv("ORACLE_API_KEY", "no-key"),
)
LOCAL_HEAVY_CLIENT = _make_client(
    os.getenv("LOCAL_HEAVY_API_BASE", "http://mac-mini.local:8000/v1"),
    os.getenv("LOCAL_HEAVY_API_KEY", "no-key"),
)

# Cloud client only if enabled
CLOUD_ENABLED = os.getenv("CLOUD_ENABLED", "false").lower() == "true"
CLOUD_CLIENT = None
if CLOUD_ENABLED:
    CLOUD_CLIENT = _make_client(
        os.getenv("CLOUD_API_BASE", "https://openrouter.ai/api/v1"),
        os.getenv("OPENROUTER_API_KEY", "no-key"),
    )

CLIENT_MAP = {
    "self":        ORACLE_CLIENT,
    "local_heavy": LOCAL_HEAVY_CLIENT,
    "cloud":       CLOUD_CLIENT,
}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Inferencacle Oracle — Intelligent Inference Router")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


async def _stream_upstream(client: AsyncOpenAI, model: str, messages: list, max_tokens: int):
    """Yield SSE chunks from upstream, forwarding them directly to the edge client."""
    stream = await client.chat.completions.create(
        model=model,
        messages=[m.model_dump() for m in messages],
        max_tokens=max_tokens,
        stream=True,
    )
    
    async for chunk in stream:
        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
            delta = chunk.choices[0].delta.content
            yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(req: OracleRequest, request: Request):
    # 1. Redact PII
    redacted = redact(req.messages)
    log.info(f"PII detected: {redacted.pii_detected}")

    # 2. Route — fast-path first, then ask the oracle model if needed
    decision = route_fast(redacted.messages, pii_detected=redacted.pii_detected)
    if decision is None:
        decision = await route_intelligent(redacted.messages, ORACLE_CLIENT, pii_detected=redacted.pii_detected)
    log.info(f"Routing → {decision.target} / {decision.model} ({decision.reason})")

    # 3. Inject oracle system prompt for self-handled requests
    messages = redacted.messages
    if decision.target == "self" and ORACLE_SYSTEM_PROMPT:
        non_system = [m for m in messages if m.role != "system"]
        messages = [ChatMessage(role="system", content=ORACLE_SYSTEM_PROMPT)] + non_system

    # 4. Optionally compress before cloud
    if decision.target == "cloud" and CLOUD_ENABLED:
        messages = await compress(messages, ORACLE_CLIENT)

    # 5. Pick client
    client = CLIENT_MAP[decision.target]
    if client is None:
        return JSONResponse(
            status_code=503,
            content={"error": f"Target '{decision.target}' is not available (cloud disabled?)"}
        )

    # 5. Stream response back
    if req.stream:
        return StreamingResponse(
            _stream_upstream(client, decision.model, messages, req.max_tokens),
            media_type="text/event-stream",
            headers={
                "X-Oracle-Route": decision.target,
                "X-Oracle-Model": decision.model,
                "X-Oracle-Tokens": str(decision.estimated_tokens),
            },
        )
    else:
        resp = await client.chat.completions.create(
            model=decision.model,
            messages=[m.model_dump() for m in messages],
            max_tokens=req.max_tokens,
        )
        return JSONResponse(
            content=resp.model_dump(),
            headers={
                "X-Oracle-Route": decision.target,
                "X-Oracle-Model": decision.model,
                "X-Oracle-Tokens": str(decision.estimated_tokens),
            },
        )


@app.get("/health")
async def health():
    return {"status": "ok", "cloud_enabled": CLOUD_ENABLED}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=int(os.getenv("ORACLE_PORT", 9000)),
        reload=True
    )
