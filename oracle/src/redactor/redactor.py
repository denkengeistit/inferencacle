"""
PII redaction using Microsoft Presidio.
Strips sensitive entities before any text leaves the oracle boundary.
"""
import os
import sys
from pathlib import Path
from typing import Tuple

# Add shared to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from shared.schemas.messages import ChatMessage, RedactedPayload

ENTITIES = os.getenv("REDACTION_ENTITIES", "PERSON,EMAIL_ADDRESS,PHONE_NUMBER").split(",")
ENABLED  = os.getenv("REDACTION_ENABLED", "true").lower() == "true"

_analyzer  = None
_anonymizer = None

def _init_engines():
    global _analyzer, _anonymizer
    if _analyzer is None:
        _analyzer = AnalyzerEngine()
        _anonymizer = AnonymizerEngine()

def redact(messages: list[ChatMessage]) -> RedactedPayload:
    """
    Redact PII from all user/system messages.
    Returns a RedactedPayload with cleaned messages and a reverse map.
    The reverse map lets you restore originals in the response if needed.
    """
    if not ENABLED:
        return RedactedPayload(messages=messages)

    _init_engines()
    
    counter = {"n": 0}
    reverse_map: dict[str, str] = {}
    pii_found = False

    def _replace(text: str) -> str:
        results = _analyzer.analyze(text=text, language="en", entities=ENTITIES)
        if not results:
            return text
        nonlocal pii_found
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
        if msg.role in ("user", "system"):
            cleaned.append(ChatMessage(role=msg.role, content=_replace(msg.content)))
        else:
            cleaned.append(msg)

    return RedactedPayload(messages=cleaned, redaction_map=reverse_map, pii_detected=pii_found)
