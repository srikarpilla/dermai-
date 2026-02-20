#!/bin/bash

echo "==> Downloading NLTK data..."
python -c "
import nltk
import ssl
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
"

echo "==> Starting Flask backend on port 5000 (model loads on first request)..."
gunicorn backend:app \
  --bind 0.0.0.0:5000 \
  --timeout 300 \
  --workers 1 \
  --log-level info &

echo "==> Starting FastAPI chatbot on port 8000..."
uvicorn derma_chat:app \
  --host 0.0.0.0 \
  --port 8000 \
  --log-level info &

echo "==> Waiting 10s for servers to bind ports..."
sleep 10

echo "==> Starting proxy on port ${PORT:-10000}..."
exec uvicorn proxy:app \
  --host 0.0.0.0 \
  --port ${PORT:-10000} \
  --log-level info
