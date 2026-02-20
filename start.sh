#!/bin/bash

# Download NLTK data needed by derma_chat.py
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# Start Flask backend on port 5000 in background
echo "Starting Flask backend on port 5000..."
gunicorn backend:app --bind 0.0.0.0:5000 --timeout 120 --workers 1 &

# Start FastAPI chatbot on port 8000 in background  
echo "Starting FastAPI chatbot on port 8000..."
uvicorn derma_chat:app --host 0.0.0.0 --port 8000 &

# Start a simple proxy on the main PORT that Render expects
# This routes / to Flask (5000) and /chat* to FastAPI (8000)
echo "Starting main proxy on port $PORT..."
uvicorn proxy:app --host 0.0.0.0 --port ${PORT:-10000}
