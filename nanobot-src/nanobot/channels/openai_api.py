"""
OpenAI-compatible API channel.

Exposes the nanobot agent as an OpenAI-compatible inference endpoint so that
external clients (Open WebUI, Chatbox, edge devices, other agents) can talk
to it via standard /v1/chat/completions and /v1/models endpoints.

This is *not* a provider that nanobot uses — it's a channel that lets
nanobot act AS a provider to the outside world.
"""

import asyncio
import re
import time
import uuid
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel

# FastAPI / uvicorn are optional deps — import lazily in start()
_app = None


class OpenAIAPIChannel(BaseChannel):
    """
    Serves an OpenAI-compatible HTTP API backed by the nanobot agent loop.

    Incoming HTTP requests are published to the message bus as InboundMessages.
    The channel waits for the corresponding OutboundMessage and returns it
    as the HTTP response.
    """

    name = "openai_api"
    display_name = "OpenAI API"

    def __init__(self, config: Any, bus: MessageBus):
        super().__init__(config, bus)
        self._port = int(getattr(config, "port", None) or config.get("port", 9000) if isinstance(config, dict) else 9000)
        self._host = "0.0.0.0"
        # Pending requests: request_id -> asyncio.Future[str]
        self._pending: dict[str, asyncio.Future] = {}
        self._server = None

    async def start(self) -> None:
        """Start the FastAPI server."""
        import uvicorn
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse
        from fastapi.middleware.cors import CORSMiddleware

        app = FastAPI(title="Inferencacle — OpenAI-compatible API")
        app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

        channel = self  # capture for closures

        @app.get("/v1/models")
        async def list_models():
            """Advertise the agent as a model with vision + tool use."""
            now = int(time.time())
            return {
                "object": "list",
                "data": [
                    {
                        "id": "inferencacle",
                        "object": "model",
                        "created": now,
                        "owned_by": "inferencacle",
                        "capabilities": {
                            "vision": True,
                            "tool_use": True,
                            "thinking": True,
                        },
                    },
                ],
            }

        @app.get("/v1/models/{model_id}")
        async def get_model(model_id: str):
            return {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "inferencacle",
            }

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            body = await request.json()
            messages = body.get("messages", [])
            stream = body.get("stream", False)

            # Extract the user's message (last user message)
            user_content = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        user_content = content
                    elif isinstance(content, list):
                        # Multimodal: extract text parts
                        user_content = " ".join(
                            part.get("text", "") for part in content
                            if part.get("type") == "text"
                        )
                    break

            if not user_content.strip():
                return JSONResponse(
                    status_code=400,
                    content={"error": {"message": "No user message found", "type": "invalid_request_error"}},
                )

            request_id = uuid.uuid4().hex
            chat_id = f"api-{request_id}"

            # Create a future to wait for the agent's response
            future: asyncio.Future[str] = asyncio.get_event_loop().create_future()
            channel._pending[request_id] = future

            # Publish to the bus — the agent loop will pick this up
            await channel._handle_message(
                sender_id="api-client",
                chat_id=chat_id,
                content=user_content,
                metadata={"request_id": request_id},
                session_key=f"openai_api:{chat_id}",
            )

            # Wait for the agent to respond (timeout after 120s)
            try:
                response_text = await asyncio.wait_for(future, timeout=120.0)
            except asyncio.TimeoutError:
                channel._pending.pop(request_id, None)
                return JSONResponse(
                    status_code=504,
                    content={"error": {"message": "Agent response timed out", "type": "timeout_error"}},
                )

            completion_id = f"chatcmpl-{request_id[:12]}"
            created = int(time.time())

            if stream:
                # SSE streaming — send the full response as chunks
                import json as _json

                async def generate_sse():
                    # Send content in a single chunk (agent doesn't stream internally)
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": "inferencacle",
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": response_text},
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {_json.dumps(chunk)}\n\n"

                    # Send done chunk
                    done_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": "inferencacle",
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }],
                    }
                    yield f"data: {_json.dumps(done_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    generate_sse(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )

            # Non-streaming response
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": "inferencacle",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        self._running = True
        logger.info("Starting OpenAI API channel on {}:{}", self._host, self._port)

        config = uvicorn.Config(app, host=self._host, port=self._port, log_level="info")
        self._server = uvicorn.Server(config)
        await self._server.serve()

    async def stop(self) -> None:
        """Stop the FastAPI server."""
        self._running = False
        if self._server:
            self._server.should_exit = True
        # Cancel any pending requests
        for future in self._pending.values():
            if not future.done():
                future.cancel()
        self._pending.clear()

    async def send(self, msg: OutboundMessage) -> None:
        """
        Receive an outbound message from the agent and resolve the
        corresponding pending HTTP request.
        """
        # Find the matching request by chat_id
        request_id = None
        for rid, future in self._pending.items():
            if msg.chat_id == f"api-{rid}":
                request_id = rid
                break

        if request_id and request_id in self._pending:
            future = self._pending.pop(request_id)
            if not future.done():
                # Skip progress messages
                if msg.metadata.get("_progress"):
                    # Re-add for the real response
                    self._pending[request_id] = future
                    return
                # Strip model control tokens and tool-call markup
                clean = msg.content or ''
                clean = re.sub(r'<\|[^|]*\|>', '', clean)  # GLM control tokens
                clean = re.sub(r'<arg_key>.*?</arg_key>', '', clean)  # tool arg keys
                clean = re.sub(r'<arg_value>.*?</arg_value>', '', clean, flags=re.DOTALL)  # tool arg values
                clean = re.sub(r'\bmessage\b\s*$', '', clean, flags=re.MULTILINE)  # bare "message" tool name
                clean = clean.strip()
                future.set_result(clean)
        else:
            logger.debug("OpenAI API channel: no pending request for chat_id={}", msg.chat_id)

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {
            "enabled": False,
            "port": 9000,
            "allowFrom": ["*"],
        }
