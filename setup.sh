#!/usr/bin/env bash
# One-shot setup for the default (Ollama + Chroma) path.
# Usage: ./setup.sh
set -euo pipefail

echo "==> Creating virtualenv (.venv) and installing dependencies"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt

echo "==> Seeding config files"
[ -f .env ] || { cp .env.example .env && echo "  created .env (edit ORGANIZATION_NAME)"; }
[ -f sources.txt ] || { cp sources.txt.example sources.txt && echo "  created sources.txt (add your URLs)"; }

if command -v ollama >/dev/null 2>&1; then
  echo "==> Pulling default Ollama models"
  ollama pull qwen2.5:0.5b-instruct
  ollama pull nomic-embed-text
else
  echo "==> Ollama not found. Install from https://ollama.com then run:"
  echo "     ollama pull qwen2.5:0.5b-instruct && ollama pull nomic-embed-text"
  echo "   (Skip if you're using a cloud provider — set LLM_PROVIDER in .env.)"
fi

echo ""
echo "Done. Next:"
echo "  1. Edit .env (ORGANIZATION_NAME) and sources.txt (your URLs)"
echo "  2. make ingest      # build the index"
echo "  3. make chat        # talk to it in the terminal"
