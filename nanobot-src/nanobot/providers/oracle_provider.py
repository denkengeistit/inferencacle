"""
Oracle provider — wraps intelligent routing and PII redaction into
the nanobot provider interface.

When the agent loop calls chat(), this provider:
1. Redacts PII from user/system messages
2. Asks the oracle model whether to handle locally or escalate
3. Forwards to the appropriate backend (local oracle or heavy model)
4. Returns a standard LLMResponse
"""

from __future__ import annotations

import uuid
from typing import Any

import json_repair
from loguru import logger
from openai import AsyncOpenAI

from nanobot.oracle.redactor import redact_messages
from nanobot.oracle.router import route_fast, route_intelligent
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class OracleProvider(LLMProvider):
    """
    LLM provider with intelligent routing between local oracle and heavy backend.
    """

    def __init__(
        self,
        local_model: str = "qwen3-vl:8b",
        local_api_base: str = "http://localhost:11434/v1",
        local_api_key: str = "no-key",
        heavy_model: str = "zai-org/glm-4.7-flash",
        heavy_api_base: str = "http://localhost:8000/v1",
        heavy_api_key: str = "no-key",
        complexity_threshold: int = 800,
        redaction_enabled: bool = True,
        redaction_entities: str = "PERSON,EMAIL_ADDRESS,PHONE_NUMBER,CREDIT_CARD,US_SSN",
    ):
        super().__init__(api_key=local_api_key, api_base=local_api_base)
        self.local_model = local_model
        self.heavy_model = heavy_model
        self.complexity_threshold = complexity_threshold
        self.redaction_enabled = redaction_enabled
        self.redaction_entities = redaction_entities.split(",") if isinstance(redaction_entities, str) else redaction_entities

        self._local_client = AsyncOpenAI(
            api_key=local_api_key,
            base_url=local_api_base,
            default_headers={"x-session-affinity": uuid.uuid4().hex},
        )
        self._heavy_client = AsyncOpenAI(
            api_key=heavy_api_key,
            base_url=heavy_api_base,
            default_headers={"x-session-affinity": uuid.uuid4().hex},
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        # 1. Redact PII
        redaction = redact_messages(
            messages,
            entities=self.redaction_entities,
            enabled=self.redaction_enabled,
        )
        if redaction.pii_detected:
            logger.info("PII detected and redacted")
        work_messages = redaction.messages

        # 2. Route — fast-path, then intelligent
        decision = route_fast(
            work_messages,
            pii_detected=redaction.pii_detected,
            complexity_threshold=self.complexity_threshold,
        )
        if decision is None:
            decision = await route_intelligent(
                work_messages,
                self._local_client,
                self.local_model,
                pii_detected=redaction.pii_detected,
            )

        # 3. Pick client and model
        if decision.escalate:
            client = self._heavy_client
            target_model = model or self.heavy_model
            logger.info("Oracle routing → heavy ({}: {})", target_model, decision.reason)
        else:
            client = self._local_client
            target_model = model or self.local_model
            logger.info("Oracle routing → self ({}: {})", target_model, decision.reason)

        # 4. Forward to backend
        kwargs: dict[str, Any] = {
            "model": target_model,
            "messages": self._sanitize_empty_content(work_messages),
            "max_tokens": max(1, max_tokens),
            "temperature": temperature,
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if tools:
            kwargs.update(tools=tools, tool_choice=tool_choice or "auto")

        try:
            return self._parse(await client.chat.completions.create(**kwargs))
        except Exception as e:
            return LLMResponse(content=f"Error: {e}", finish_reason="error")

    def _parse(self, response: Any) -> LLMResponse:
        choice = response.choices[0]
        msg = choice.message
        tool_calls = [
            ToolCallRequest(
                id=tc.id,
                name=tc.function.name,
                arguments=json_repair.loads(tc.function.arguments)
                if isinstance(tc.function.arguments, str)
                else tc.function.arguments,
            )
            for tc in (msg.tool_calls or [])
        ]
        u = response.usage
        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage={
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
            } if u else {},
            reasoning_content=getattr(msg, "reasoning_content", None) or None,
        )

    def get_default_model(self) -> str:
        return self.local_model
