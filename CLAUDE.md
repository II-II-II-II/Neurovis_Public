# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Neurovis is a personal biofeedback/neurofeedback platform combining real-time EEG (Muse headband) + HRV (Polar H10 chest strap / Apple Watch) sensor data with AI-driven analysis. There is no single app — the repo contains three loosely related subsystems that evolved over time. There is no build system, package manifest, linter, or test suite; everything is run directly with `python3`.

**Mission/positioning** (per neurovis.io, the live site — hosted on the author's own Raspberry Pi 4 at home behind Cloudflare): the product's core promise is user-owned, locally-processed biometric data — "Your Biometrics. Unlocked & Uncompromised" — explicitly rejecting centralized/subscription health platforms in favor of edge computing where data "stays entirely in your control." The `Fusion/` in-browser Pyodide architecture is a direct expression of that promise (real processing happens client-side, nothing round-trips to a server by default). Keep this in mind when reviewing changes: anything that quietly centralizes data, shares state across users/sessions, or relies on a "security" mechanism whose secret is visible client-side runs directly counter to the product's stated purpose and should be flagged, not just noted as a style nit.

## Repository layout

- **`Agents/`** — the current, actively developed stack: a FastAPI backend with a router agent that delegates to two specialist LlamaIndex ReAct agents, plus the HTML chat UI. This is the primary place to work unless told otherwise.
- **`Fusion/`** — a separate, self-contained real-time dashboard that runs entirely in the browser (Web Bluetooth + Pyodide/WASM Python). No Python backend process is involved at runtime.
- **`HRV/`** — an earlier, standalone iteration of the HRV agent plus a small Flask ingestion endpoint for a companion iOS/shortcut upload flow. `HRV/HRVAgent.py` and `HRV/HRVAgent.html` are byte-identical to their counterparts in `Agents/` (kept from before the router was introduced) — treat `Agents/` as the source of truth and only touch `HRV/`'s copies if explicitly asked to keep the standalone version in sync.

Everything logs verbose agent "thoughts" to `neurovis_agent_thoughts.log` (created at runtime, not committed) — check this file first when debugging agent tool-selection or routing behavior.

## Running things

No dependency manifest exists; install what a given script imports (`fastapi`, `uvicorn`, `pandas`, `numpy`, `scipy`, `matplotlib`, `pydantic`, `llama-index-core`, `llama-index-llms-ollama`, `llama-index-embeddings-huggingface`, `flask`, `flask-cors`). The LLM stack expects a **local Ollama server** running the `qwen2.5` model (`ollama pull qwen2.5`) — nothing in `Agents/` or `HRV/` calls a cloud LLM API.

```bash
# Primary stack: router + both specialist agents, chat UI on http://localhost:8000
cd Agents && python3 neurovisAgent.py

# Standalone HRV-only agent (also binds :8000 — do not run alongside neurovisAgent.py)
cd Agents && python3 HRVAgent.py   # or: cd HRV && python3 HRVAgent.py

# Fusion dashboard: pure static files, just needs an HTTP server (Pyodide fetches the .py files as text)
cd Fusion && python3 -m http.server 8080   # then open Neurovis.html

# HRV/ companion upload receiver (Flask, expects signed payloads from NeurovisAW.html)
cd HRV && python3 uploader.py   # binds :8002
```

There's no automated test suite. Validate changes by running the relevant server and exercising it through its HTML UI (or `curl` against `/chat` / `/upload_context`).

## Architecture: `Agents/` (primary stack)

`neurovisAgent.py` is a FastAPI app that owns a **router `ReActAgent`** (`ROUTER_PROMPT`) with exactly two tools, each a thin wrapper that hands the query to another module's own agent:

- `call_macro_biometrician` → `HRVAgent.agent` — Apple Watch data: sleep, daily HRV/HR baselines, workouts, readiness score.
- `call_neuro_analyst` → `neuroAgent.neuro_agent` — meditation session EEG data: FAA, Engagement/Flow/Vigilance/Detachment, per-session and historical trends.

Both `HRVAgent` and `neuroAgent` are plain Python modules imported directly (not subprocesses/microservices) and each keeps its own module-level pandas DataFrames as agent "memory" — there is no database. `HRVAgent.py` also has its own FastAPI app/routes so it can run standalone (see port note above); when run under the router, `neurovisAgent.py`'s startup event manually invokes `HRVAgent.startup_event()` since FastAPI won't fire a sub-app's lifecycle hooks automatically.

**Data routing** happens in `neurovisAgent.py`'s `/upload_context` endpoint, dispatched purely by filename pattern:
- `*.json` containing `raw_details` → `HRVAgent.upload_data` (Apple Watch export)
- other `*.json` → `neuroAgent.upload_context` (meditation session backup/history)
- `*.csv` without `apple` in the name → `neuroAgent.upload_context` (single-session EEG/HRV CSV)
- `*.csv` with `apple` in the name → loaded directly into `HRVAgent.df_hrv`

**Agent design pattern** used throughout (`HRVAgent.py`, `neuroAgent.py`, `neurovisAgent.py`): agents never write ad-hoc pandas/Python — every capability is a `FunctionTool` with a strict docstring contract (valid metric names, valid enum values, when *not* to call it), and the system prompt encodes explicit "guardrails" (e.g. refuse metrics not in the glossary, never guess REM HRV, silently swap/deduplicate date ranges before comparing periods, ask for clarification when a metric like "HRV" is ambiguous between data sources). When adding a new tool, follow this same pattern — a narrow, defensively-validated function plus a docstring the LLM will read literally as routing instructions — rather than giving the agent general code execution.

Both `HRVAgent.py`'s and `neuroAgent.py`'s chat endpoints inject a dynamic "schema guide" / column list into the prompt before each turn so the LLM only ever references columns that are actually loaded — preserve this if you change how data is ingested.

## Architecture: `Fusion/` (browser-native real-time dashboard)

`Neurovis.html` ("Pyodide Edge Dashboard") has no server-side logic. At load time it:
1. Boots Pyodide (Python-in-WASM) and installs numpy/pandas/scipy into it.
2. Fetches `hardware.py` and `Neurovis.py` as raw text and writes them into Pyodide's virtual filesystem, then `import`s `Neurovis` inside the WASM runtime.
3. Connects directly to Muse (EEG) and Polar H10 (HR) hardware via the **Web Bluetooth API** in `hardware.js` — there is no OSC/LSL bridge process in this path (that's legacy — see `OSC_IP`/`OSC_PORT` constants in `hardware.py`, unused in the browser flow).
4. Drives everything through `process_tick(session_id, hardware_payload, ui_command)` in `Neurovis.py`, called synchronously ~20Hz from JS — this function *is* the "backend", just executed client-side. A mock `window.ws` object with a `.send()` method fakes a WebSocket API so the surrounding UI code didn't need to change when the real WebSocket server was replaced by Pyodide.
5. Persists session history, per-minute "climate chunks", and raw waveform data to **IndexedDB** (`NeurovisDB`) in the browser — nothing is sent to a server by default.

`MindRelay.html` reads that IndexedDB history for longitudinal trend views and optionally calls the Gemini API directly from the browser using a user-supplied API key stored in `localStorage` (BYOK — no proxy/backend involved). `emotionreplay.html` is a forensic session replay/review UI over the same stored data. These three HTML files are independent entry points into the same IndexedDB store, not a multi-page app with shared routing.

## Architecture: `HRV/` (legacy/companion)

`uploader.py` is an unrelated minimal Flask receiver (port 8002) for `NeurovisAW.html`'s upload flow: it validates a shared-secret SHA-256 HMAC-style hash (`X-Payload-Hash` header vs. `payload + SECRET_SALT`), enforces a 60-second payload TTL to reject replays, and writes accepted payloads as UUID-named files under `./data_lake`. This is a separate concern from the `Agents/` upload pipeline and doesn't share code with it — don't assume changes to one affect the other.
