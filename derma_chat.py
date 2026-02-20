# derma_chat.py
import os
import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

BASE = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Dr. Derm (RAG Chat)")

# Allow CORS while developing — narrow this in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # <-- change to specific origins in production e.g. ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static assets (css/js/images) used by derma_chat.html from /static/*
static_dir = os.path.join(BASE, "static")
if not os.path.exists(static_dir):
    # create folder if missing so mount doesn't fail (empty is ok)
    os.makedirs(static_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Data models for chat API
class ChatRequest(BaseModel):
    query: str
    history: Optional[List[dict]] = None


@app.get("/chat-ui", response_class=HTMLResponse)
async def chat_ui():
    """
    Serve the chat page. This is the key fix: previous code only had '/' while UI links hit '/chat-ui'.
    """
    html_path = os.path.join(BASE, "derma_chat.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content)
    return HTMLResponse("<h1>Chat UI not found</h1>", status_code=404)


@app.get("/", response_class=HTMLResponse)
async def root():
    # Optional: redirect or show a small message
    return HTMLResponse("<h1>Dr. Derm Chat API is running</h1><p>Open /chat-ui</p>")


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    """
    Handle a chat request from the frontend. Integrate with your RAG functions here:
    - prefer calling your existing `find_best_matches(query)` and `derma_answer(query, matches)` (or similarly named)
    - if those functions exist elsewhere in your project, import and call them below.
    """
    q = req.query

    # Try to call existing functions if present in project
    try:
        # If you have functions in a module named rag or derma_rag, import them here:
        # from rag_module import find_best_matches, derma_answer
        # matches = find_best_matches(q)
        # answer = derma_answer(q, matches)
        # For safety fallback, only call if available
        matches = None
        answer = None
        # attempt import dynamically (no error if not present)
        import importlib

        if importlib.util.find_spec("derma_rag") is not None:
            rg = importlib.import_module("derma_rag")
            if hasattr(rg, "find_best_matches"):
                matches = rg.find_best_matches(q)
            if hasattr(rg, "derma_answer"):
                answer = rg.derma_answer(q, matches)
        elif importlib.util.find_spec("derma_chat_helpers") is not None:
            rg = importlib.import_module("derma_chat_helpers")
            if hasattr(rg, "find_best_matches"):
                matches = rg.find_best_matches(q)
            if hasattr(rg, "derma_answer"):
                answer = rg.derma_answer(q, matches)

    except Exception as ex:
        # log or print in real app
        print("RAG call error:", ex)
        matches = None
        answer = None

    # fallback simple response if RAG not wired
    if not answer:
        # Example simple placeholder: echo + advice
        answer = f"I got your question: '{q}'. I couldn't find the RAG module, so this is a fallback response. Please check logs or add your RAG functions."

    return JSONResponse({"answer": answer, "matches": matches})


# Optional health endpoint
@app.get("/health")
async def health():
    return {"status": "ok"}
