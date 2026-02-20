#!/bin/bash
set -e

echo "==> Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

echo "==> Starting Flask backend on port 5000..."
gunicorn backend:app \
  --bind 0.0.0.0:5000 \
  --timeout 120 \
  --workers 1 \
  --log-level info &
FLASK_PID=$!

echo "==> Starting FastAPI chatbot on port 8000..."
uvicorn derma_chat:app \
  --host 0.0.0.0 \
  --port 8000 \
  --log-level info &
CHAT_PID=$!

echo "==> Waiting for Flask to be ready..."
for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:5000/ > /dev/null 2>&1; then
    echo "==> Flask is ready!"
    break
  fi
  echo "    Flask not ready yet, waiting... ($i/60)"
  sleep 3
done

echo "==> Waiting for FastAPI to be ready..."
for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "==> FastAPI is ready!"
    break
  fi
  echo "    FastAPI not ready yet, waiting... ($i/30)"
  sleep 2
done

echo "==> Both servers ready. Starting proxy on port ${PORT:-10000}..."
uvicorn proxy:app \
  --host 0.0.0.0 \
  --port ${PORT:-10000} \
  --log-level info

# If proxy dies, kill background servers too
kill $FLASK_PID $CHAT_PID 2>/dev/null
