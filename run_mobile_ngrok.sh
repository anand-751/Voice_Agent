#!/usr/bin/env bash
set -e

echo "============================================================"
echo "  Bright Dental Clinic — Mobile Voice Agent Launcher (ngrok)"
echo "============================================================"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# 1. Build frontend bundle
echo "📦 [1/3] Building frontend bundle..."
cd frontend && npm run build && cd "$DIR"

# 2. Start Uvicorn backend server on port 8000 in background
echo "🚀 [2/3] Starting backend server on http://0.0.0.0:8000..."
PYTHONPATH=backend ./.venv/bin/python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

cleanup() {
    echo ""
    echo "🛑 Shutting down server and tunnel..."
    kill $SERVER_PID 2>/dev/null || true
    exit 0
}
trap cleanup INT TERM EXIT

# Wait for server to become healthy
echo "⏳ Waiting for backend to start..."
sleep 2

# 3. Launch ngrok on port 8000
echo "🌐 [3/3] Launching ngrok tunnel on port 8000..."
echo "============================================================"
echo "  👉 Open the public HTTPS link on your mobile phone! 📱"
echo "  👉 Press Ctrl+C anytime to stop."
echo "============================================================"
echo ""

ngrok http 8000
