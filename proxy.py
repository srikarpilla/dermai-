"""
proxy.py — Unified entry point for Render
Routes:
  /predict, /         → Flask backend (port 5000)
  /chat, /health,
  /suggest, /chat-ui  → FastAPI chatbot (port 8000)
"""

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FLASK_URL  = "http://127.0.0.1:5000"
CHAT_URL   = "http://127.0.0.1:8000"

# Routes that go to the FastAPI chatbot
CHAT_ROUTES = ["/chat", "/suggest", "/health", "/chat-ui", "/docs", "/openapi.json"]


async def forward(request: Request, base_url: str) -> Response:
    url = base_url + request.url.path
    if request.url.query:
        url += "?" + request.url.query

    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.request(
            method  = request.method,
            url     = url,
            headers = headers,
            content = body,
        )

    return Response(
        content     = resp.content,
        status_code = resp.status_code,
        headers     = dict(resp.headers),
        media_type  = resp.headers.get("content-type"),
    )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def route_all(request: Request, path: str):
    full_path = "/" + path

    # Send chat/AI routes to FastAPI chatbot
    for prefix in CHAT_ROUTES:
        if full_path.startswith(prefix):
            return await forward(request, CHAT_URL)

    # Everything else goes to Flask
    return await forward(request, FLASK_URL)
