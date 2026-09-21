# Setup and configuration

Forma runs embeddings and language-model inference locally. The default launcher is configured for the GGUF already present on the development machine; a fresh clone must point it at a compatible local model and llama.cpp executable.

## Prerequisites and startup

Install Python 3.11+, Node 20.19+, Ollama, and a compatible `llama-server`. The development environment uses Python 3.14, Node 25, and an Apple Silicon Mac. Keep Ollama running.

```sh
export MODEL_PATH=/absolute/path/to/model.gguf
export LLAMA_SERVER=/absolute/path/to/llama-server
npm run setup
npm run dev
```

Setup installs backend/frontend dependencies, seeds the synthetic SQLite database, downloads `all-minilm:22m` through Ollama, and builds the Chroma index. It does not download the generation model. Subsequent launches reuse persisted data and vectors.

The app opens at **http://127.0.0.1:5178**. The launcher starts the model on **8091**, FastAPI on **8008**, and Vite on **5178**. It reuses a model already healthy on 8091. Ctrl+C stops processes started by the launcher and leaves a separately started model running. Fonts are bundled locally.

You can also run `npm run model`, `npm run api`, and `npm run web` in separate terminals. For a production frontend build, run `npm run build`; the API then serves `frontend/dist` at **http://127.0.0.1:8008**. Use one API worker with the embedded Chroma database.

## Model configuration

The default model path is `~/UltimateDocumentEditor/Qwen3.8-27B-UD-IQ1_S.gguf`. The launcher first looks for the existing bundled b9150 runtime at `~/UltimateDocumentEditor/desktop/resources/darwin-arm64/runtime/llama-b9150/llama-server`, then for `llama-server` on PATH. `MODEL_PATH` and `LLAMA_SERVER` override these defaults.

The inspected GGUF reports `qwen35` architecture, approximately 26.90 billion parameters, and IQ1_S quantization. Its 5.77 GiB weights were tested on a 16 GiB Apple M5. The filename and metadata are reported as found, rather than an independent certification of model lineage or a claim of benchmark superiority. Quantization and hardware affect response quality and latency.

The launcher uses an 8,192-token context, one inference slot, Metal/GPU offload where supported, and 512/256 logical/physical batch sizes. The launcher and backend disable thinking and request separate reasoning fields. History is sanitized before templating; `preserve_thinking: true` keeps Qwen's empty reasoning separators on past assistant turns, with no actual reasoning retained. The backend checks the actual token budget and validates responses before display. A different model/runtime may need compatible launcher flags or a different chat-template setting.

| Variable | Default / purpose |
| --- | --- |
| `MODEL_PATH` | Existing GGUF path described above |
| `LLAMA_SERVER` | Bundled runtime, otherwise executable from PATH |
| `MODEL_URL` | `http://127.0.0.1:8091` |
| `MODEL_NAME` | `qwen-local-27b`, the model alias sent to inference |
| `MODEL_LABEL` | `Qwen3.8 · 27B`, displayed in the UI |
| `MODEL_BATCH`, `MODEL_UBATCH` | `512`, `256` |
| `EMBEDDING_MODEL` | `all-minilm:22m` |
| `EMBEDDING_URL` | `http://127.0.0.1:11434` |
| `DATABASE_PATH` | `data/students.sqlite3` |
| `CHROMA_PATH` | `data/chroma`; takes precedence over legacy `VECTOR_PATH` |

Export variables in the shell; environment files are not loaded automatically. `MODEL_PORT` changes the model launcher's port, but the combined development launcher checks port 8091. For a non-default model port, run services separately and set `MODEL_URL` to match. The inference server must implement `/health`, `/props`, `/apply-template`, `/tokenize`, and `/v1/chat/completions` (the llama.cpp server API).

## Data and checks

`python -m backend.db` seeds a fresh database or applies the additive version-2 expansion and version-3 synthetic timing migration without duplicating records or replacing existing observations. The timing migration adds `time_taken_seconds` and regenerates duration-aware evidence. The next vector-index initialization rebuilds changed embeddings automatically. If records are intentionally changed, restart the API to rebuild evidence documents, or call `backend.db.build_index(connection)` and commit. Chroma detects the changed evidence fingerprint. With the API stopped, `npm run vectors` builds or verifies the index without concurrent index writers.

```sh
npm test
npm run build
node frontend/check-ui.mjs
node frontend/check-chat.mjs
node frontend/check-insights.mjs
node frontend/check-timing.mjs
node frontend/check-conversation.mjs
.venv/bin/python -m scripts.check_conversation
.venv/bin/python -m scripts.check_followups
```

Browser checks require the running app and installed Chrome. Chat and insight checks use the real local model; backend unit tests use deterministic embeddings and simulated generation. `.venv/bin/python scripts/benchmark.py` measures live first-text and total response times, including a repeated request to exercise caching. Browser screenshots and benchmark results are written under `/private/tmp/forma-*`.

## Temporary sharing

```sh
cloudflared tunnel --url http://localhost:5178
```

The Vite configuration includes the exact hostname allowed for the development session's tunnel. When Cloudflare issues a different URL, replace that hostname in `frontend/vite.config.ts` with the new hostname. The frontend proxies `/api` to FastAPI. Tunnel URLs depend on the running local services and tunnel process; no permanent deployment is included. The application has no authentication, and only synthetic records are intended for this demo.
