# inferencacle

**inference + oracle** — A 4-layer intelligent inference architecture with privacy-aware routing, PII redaction, and prompt optimization.

## Architecture

**Layer 1** (Edge): STT/TTS + vision on device (iPhone/MacBook Neo)  
**Layer 2** (Oracle): Local routing, PII redaction, prompt coaching (iMac M4 32GB)  
**Layer 3** (Heavy Local): Large model inference (Mac Mini M4 Pro 48GB)  
**Layer 4** (Cloud): Optional cloud escalation (disabled by default)

## Components

- `edge-client/` — Realtime voice chat interface
- `oracle/` — FastAPI proxy with routing, redaction, and compression
- `shared/` — Pydantic schemas shared across layers
- `config/` — Configuration templates
- `scripts/` — Utilities and deployment helpers

## Quick Start

See `IMPLEMENTATION_PLAN.md` for full setup instructions.
