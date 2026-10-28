#!/bin/bash
# Local chat GUI for Agora (and other Ollama models): Open WebUI on http://localhost:8080
# Single-user (no login), Ollama at localhost:11434, free web search via DuckDuckGo (toggle "Web Search" in the chat box).
# Data (chats, settings) lives in ~/.open-webui. Stop with Ctrl-C (or: pkill -f "open-webui serve").
export DATA_DIR="$HOME/.open-webui"
export WEBUI_AUTH=False
export OLLAMA_BASE_URL=http://localhost:11434
export ENABLE_WEB_SEARCH=True ENABLE_RAG_WEB_SEARCH=True
export WEB_SEARCH_ENGINE=duckduckgo RAG_WEB_SEARCH_ENGINE=duckduckgo
export WEB_SEARCH_RESULT_COUNT=5 RAG_WEB_SEARCH_RESULT_COUNT=5
export ENABLE_OPENAI_API=False
export ANONYMIZED_TELEMETRY=False DO_NOT_TRACK=True SCARF_NO_ANALYTICS=True
mkdir -p "$DATA_DIR"
exec uvx --python 3.11 open-webui@latest serve --host 127.0.0.1 --port 8080
