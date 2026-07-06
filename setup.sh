#!/usr/bin/env bash
# Installs the Python environment and seeds config files. Does not provision a
# model backend (Ollama, OpenAI, etc.) — see README.md > Manual setup for that.
# Usage: ./setup.sh
set -euo pipefail

echo "==> Creating virtualenv (.venv) and installing dependencies"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt

echo "==> Seeding secrets file"
[ -f .env ] || { cp .env.example .env && echo "  created .env (fill in secrets for your chosen provider, if any)"; }

echo ""
echo "Done. This script does not provision a model backend — that step is"
echo "provider-specific. See README.md > Manual setup > Model backend and"
echo "Swapping providers for what your chosen LLM_PROVIDER / EMBEDDING_PROVIDER"
echo "requires (a local server + pulled weights, or an API key)."
echo ""
echo "Next:"
echo "  1. Edit config.py (ORGANIZATION_NAME, provider selection, etc.)"
echo "  2. Fill in .env with any provider secrets (API keys)"
echo "  3. Create sources.txt and list your source URLs, one per line"
echo "     (see sources.txt.example for the format)"
echo "  4. Provision your model backend (see above)"
echo "  5. make ingest      # build the index"
echo "  6. make chat        # talk to it in the terminal"
