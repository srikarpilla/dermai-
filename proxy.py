"""
proxy.py — Unified entry point for Render
Routes:
  /predict, /health-flask, static files → Flask (port 5000)
  /chat, /suggest, /health, /chat-ui   → FastAPI (port 8000)
"""

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FLASK_URL = "http://127.0.0.1:5000"
CHAT_URL  = "http://127.0.0.1:8000"

# Paths handled by FastAPI chatbot
CHAT_PREFIXES = ["/chat", "/suggest", "/health", "/chat-ui", "/docs", "/openapi.json"]

LOADING_HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="8">
  <title>DermAI — Starting up...</title>
  <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    body {
      font-family: 'Segoe UI', sans-serif;
      background: #f0fdfa;
      display: flex; align-items: center; justify-content: center;
      min-height: 100vh; flex-direction: column; gap: 20px;
    }
    .card {
      background: white; border-radius: 20px;
      padding: 48px 56px; text-align: center;
      box-shadow: 0 8px 40px rgba(13,148,136,0.15);
      max-width: 480px;
    }
    .logo { font-size: 48px; margin-bottom: 16px; }
    h1 { font-size: 24px; color: #0f172a; margin-bottom: 10px; }
    p  { color: #64748b; font-size: 15px; line-height: 1.6; }
    .spinner {
      width: 48px; height: 48px; border: 4px solid #ccfbf1;
      border-top-color: #0d9488; border-radius: 50%;
      animation: spin 0.9s linear infinite;
      margin: 24px auto 0;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .note { font-size: 13px; color: #94a3b8; margin-top: 16px; }
  </style>
</head>
<body>
  <div class="card">
    <div class="logo">🩺</div>
    <h1>DermAI is starting up...</h1>
    <p>The AI model is loading. This takes about <strong>60 seconds</strong> on first start. Page will refresh automatically.</p>
    <div class="spinner"></div>
    <p class="note">This delay only happens after 15 min of inactivity (free tier sleep mode).</p>
  </div>
</body>
</html>
"""


async def forward(request: Request, base_url: str) -> Response:
    url = base_url + request.url.path
    if request.url.query:
        url += "?" + request.url.query

    body    = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.request(
            method  = request.method,
            url     = url,
            headers = headers,
            content = body,
        )

    return Response(
        content    = resp.content,
        status_code= resp.status_code,
        headers    = dict(resp.headers),
        media_type = resp.headers.get("content-type"),
    )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def route_all(request: Request, path: str):
    full_path = "/" + path

    # Route to FastAPI chatbot
    for prefix in CHAT_PREFIXES:
        if full_path.startswith(prefix):
            try:
                return await forward(request, CHAT_URL)
            except Exception:
                return JSONResponse({"error": "Chatbot not ready yet, please retry."}, status_code=503)

    # Route to Flask — show loading page if not ready yet
    try:
        return await forward(request, FLASK_URL)
    except httpx.ConnectError:
        # Flask still loading model — show friendly loading page
        if request.method == "POST":
            return JSONResponse(
                {"error": "Model is still loading, please wait 30 seconds and try again."},
                status_code=503
            )
        return HTMLResponse(LOADING_HTML, status_code=503)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)
