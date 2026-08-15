#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"

echo ""
echo "  ╔══════════════════════════════════════╗"
echo "  ║       QwenChat — Local AI Bot        ║"
echo "  ║   Qwen2.5-1.5B + LoRA Fine-Tune      ║"
echo "  ╚══════════════════════════════════════╝"
echo ""

# ── Check Python ───────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo "  ❌  python3 not found. Please install Python 3.9+."
  exit 1
fi

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  ✓  Python $PY_VER found"

# ── Virtual env ────────────────────────────────────────
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "  ⟳  Creating virtual environment…"
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
echo "  ✓  Virtual environment active"

# ── Install deps ───────────────────────────────────────
echo "  ⟳  Installing / verifying dependencies…"
pip install -q --upgrade pip
pip install -q -r "$BACKEND_DIR/requirements.txt"
echo "  ✓  Dependencies ready"

# ── Start backend ──────────────────────────────────────
echo ""
echo "  🚀  Starting backend on http://localhost:8000"
echo "  📂  Model adapter: $SCRIPT_DIR/model"
echo "  🌐  Open the UI:   $SCRIPT_DIR/frontend/index.html"
echo ""
echo "  (Press Ctrl+C to stop)"
echo ""

cd "$BACKEND_DIR"
exec uvicorn server:app --host 0.0.0.0 --port 8000 --env-file ../.env --reload
