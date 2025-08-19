#!/usr/bin/env bash
set -euo pipefail

# Build a macOS .app bundle for this project using PyInstaller.
# Requires running on macOS (Darwin) with internet access to resolve tools.

if [[ "${OSTYPE:-}" != darwin* ]]; then
  echo "This script must be run on macOS (Darwin). You're on: ${OSTYPE:-unknown}" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

ARCH=$(uname -m)

# Ensure dist directories are clean
rm -rf dist build "Zotero MCP.spec" || true

# Use uvx to run pyinstaller without polluting the project deps
UVX=${UVX:-uvx}

# pyinstaller options
APP_NAME="Zotero MCP (${ARCH})"
IDENTIFIER="dev.local.pdf_mcp"

# Note: We rely on --collect-all to include data/binaries for dynamic packages.
# If packaging misses something at runtime, add another --collect-all/--hidden-import here.

"${UVX}" pyinstaller \
  --noconfirm \
  --windowed \
  --name "${APP_NAME}" \
  --osx-bundle-identifier "${IDENTIFIER}" \
  --collect-all fastapi \
  --collect-all fastmcp \
  --collect-all httpx \
  --collect-all certifi \
  --collect-all pydantic \
  --collect-all pymupdf \
  --collect-all llama_index \
  --collect-all llama_index.embeddings.ollama \
  --collect-all llama_index.node_parser.docling \
  --collect-all llama_index.readers.docling \
  --collect-all llama_index.vector_stores.milvus \
  --collect-all grpc \
  --collect-all grpcio \
  --collect-all google \
  --collect-all numpy \
  --collect-all pandas \
  --hidden-import "llama_index.node_parser.docling" \
  --hidden-import "llama_index.readers.docling" \
  --hidden-import "llama_index.embeddings.ollama" \
  --hidden-import "llama_index.vector_stores.milvus" \
  main.py

echo "Built app bundle at: ${REPO_ROOT}/dist/${APP_NAME}.app"

# Optional ad-hoc signing to reduce Gatekeeper prompts
if command -v codesign >/dev/null 2>&1; then
  echo "Ad-hoc signing app..."
  codesign --force --deep --sign - "${REPO_ROOT}/dist/${APP_NAME}.app" || true
fi

