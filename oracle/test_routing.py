#!/usr/bin/env python3
"""
Test oracle routing with different query types.
"""
import httpx
import json

BASE = "http://localhost:9000/v1/chat/completions"

tests = [
    ("short factual", "What is 2+2?"),
    ("PII trigger", "My name is Alice Johnson, email alice@example.com, SSN 123-45-6789. Can you help?"),
    ("medium complexity", "Explain the key differences between supervised and unsupervised machine learning, including at least three examples of algorithms in each category."),
    ("long complex", "Explain quantum computing " * 200),  # Will exceed token threshold
]

for label, prompt in tests:
    try:
        r = httpx.post(
            BASE,
            json={"messages": [{"role": "user", "content": prompt}], "stream": False, "max_tokens": 32},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        route = r.headers.get("x-oracle-route", "?")
        model = r.headers.get("x-oracle-model", "?")
        tokens = r.headers.get("x-oracle-tokens", "?")
        print(f"[{label:20s}] → {route:12s} | {model:20s} | {tokens:>4s} tokens")
    except Exception as e:
        print(f"[{label:20s}] → ERROR: {e}")
