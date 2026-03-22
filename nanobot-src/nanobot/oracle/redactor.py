"""
PII redaction using Microsoft Presidio.
Strips sensitive entities before any text leaves the oracle boundary.
"""

import os
from dataclasses import dataclass, field
from loguru import logger

ENTITIES = os.getenv("REDACTION_ENTITIES", "PERSON,EMAIL_ADDRESS,PHONE_NUMBER,CREDIT_CARD,US_SSN").split(",")
ENABLED = os.getenv("REDACTION_ENABLED", "true").lower() == "true"

_analyzer = None
_anonymizer = None
_presidio_available = None


def _check_presidio() -> bool:
    global _presidio_available
    if _presidio_available is None:
        try:
            import presidio_analyzer  # noqa: F401
            import presidio_anonymizer  # noqa: F401
            _presidio_available = True
        except ImportError:
            _presidio_available = False
            logger.warning("Presidio not installed — PII redaction disabled. pip install presidio-analyzer presidio-anonymizer spacy")
    return _presidio_available


def _init_engines():
    global _analyzer, _anonymizer
    if _analyzer is None:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        _analyzer = AnalyzerEngine()
        _anonymizer = AnonymizerEngine()


@dataclass
class RedactionResult:
    """Result of PII redaction on a list of messages."""
    messages: list[dict]
    redaction_map: dict[str, str] = field(default_factory=dict)
    pii_detected: bool = False


def redact_messages(messages: list[dict], entities: list[str] | None = None, enabled: bool | None = None) -> RedactionResult:
    """
    Redact PII from user/system message content.

    Works with standard OpenAI message dicts (role + content).
    Only redacts string content in user and system messages.
    Passes through everything else (tool calls, images, etc.) untouched.
    """
    if (enabled if enabled is not None else ENABLED) is False:
        return RedactionResult(messages=messages)

    if not _check_presidio():
        return RedactionResult(messages=messages)

    _init_engines()

    from presidio_anonymizer.entities import OperatorConfig

    target_entities = entities or ENTITIES
    counter = {"n": 0}
    reverse_map: dict[str, str] = {}
    pii_found = False

    def _replace(text: str) -> str:
        nonlocal pii_found
        results = _analyzer.analyze(text=text, language="en", entities=target_entities)
        if not results:
            return text
        pii_found = True
        operators = {}
        for r in results:
            placeholder = f"<{r.entity_type}_{counter['n']}>"
            counter["n"] += 1
            original = text[r.start:r.end]
            reverse_map[placeholder] = original
            operators[r.entity_type] = OperatorConfig("replace", {"new_value": placeholder})
        return _anonymizer.anonymize(text=text, analyzer_results=results, operators=operators).text

    cleaned = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role in ("user", "system") and isinstance(content, str):
            cleaned.append({**msg, "content": _replace(content)})
        elif role in ("user", "system") and isinstance(content, list):
            # Multimodal: only redact text blocks
            new_content = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    new_content.append({**part, "text": _replace(part["text"])})
                else:
                    new_content.append(part)
            cleaned.append({**msg, "content": new_content})
        else:
            cleaned.append(msg)

    return RedactionResult(messages=cleaned, redaction_map=reverse_map, pii_detected=pii_found)
