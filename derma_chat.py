"""
derma_chat.py — DermAI RAG Chatbot (FastAPI)
Mounted at /chat-ui by backend.py
Production-safe for Render
"""

import os
import ssl
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

import cohere

# ─────────────────────────────────────────────
# 1. Environment
# ─────────────────────────────────────────────
load_dotenv()

COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "").strip()
COHERE_MODEL   = os.environ.get("COHERE_MODEL", "command-r").strip()

# ⚠️ DO NOT CRASH if key missing
if not COHERE_API_KEY:
    print("WARNING: COHERE_API_KEY not set. Chatbot will be disabled.")
    co = None
else:
    co = cohere.Client(api_key=COHERE_API_KEY)

# ─────────────────────────────────────────────
# 2. NLTK Setup (Safe for Render)
# ─────────────────────────────────────────────
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

ps = PorterStemmer()
CHUNK_SIZE = 1000
TOP_N = 4

# ─────────────────────────────────────────────
# 3. FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(title="DermAI Dermatologist Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# 4. In-Memory Vector Store
# ─────────────────────────────────────────────
STATE = {
    "original_docs": [],
    "processed_docs": [],
    "vectorizer": TfidfVectorizer(),
    "vectors": None,
}

# ─────────────────────────────────────────────
# 5. Built-in Dermatology Knowledge Base
# (Shortened slightly for safety)
# ─────────────────────────────────────────────
DERMA_KNOWLEDGE = """
Acne is a common skin condition caused by excess oil, bacteria, and clogged pores.
Eczema causes itchy inflamed skin and is linked to immune dysfunction.
Psoriasis is an autoimmune disease causing red scaly patches.
Fungal infections like ringworm are caused by dermatophyte fungi.
Contact dermatitis occurs due to allergens or irritants.
Urticaria (hives) results from histamine release.
Melanoma is a serious skin cancer related to UV exposure.
Healthy skin needs hydration, sunscreen, and balanced nutrition.
"""

# ─────────────────────────────────────────────
# 6. Build Knowledge Base
# ─────────────────────────────────────────────
def chunk_text(text: str, size: int = CHUNK_SIZE) -> List[str]:
    sentences = nltk.sent_tokenize(text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) > size and current:
            chunks.append(current.strip())
            current = sent + " "
        else:
            current += sent + " "
    if current.strip():
        chunks.append(current.strip())
    return chunks


def stem_text(text: str) -> str:
    return " ".join(ps.stem(w) for w in text.lower().split())


def build_knowledge_base():
    chunks = chunk_text(DERMA_KNOWLEDGE)
    STATE["original_docs"] = chunks
    STATE["processed_docs"] = [stem_text(c) for c in chunks]
    STATE["vectorizer"] = TfidfVectorizer()
    STATE["vectors"] = STATE["vectorizer"].fit_transform(
        STATE["processed_docs"]
    )
    print(f"Knowledge base built: {len(chunks)} chunks loaded.")


build_knowledge_base()

# ─────────────────────────────────────────────
# 7. Retrieval
# ─────────────────────────────────────────────
def find_best_matches(query: str, top_n: int = TOP_N) -> List[str]:
    if STATE["vectors"] is None:
        return []

    q_vec = STATE["vectorizer"].transform([stem_text(query)])
    scores = (STATE["vectors"] * q_vec.T).toarray().flatten()
    top_idx = scores.argsort()[-top_n:][::-1]

    return [
        STATE["original_docs"][i]
        for i in top_idx
        if scores[i] > 0
    ]


# ─────────────────────────────────────────────
# 8. AI Response
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """
You are Dr. Derm, a warm and caring AI dermatologist.
Speak clearly and kindly like a real doctor.
Never give a final diagnosis.
Always recommend consulting a dermatologist.
"""


def derma_answer(query: str, context_chunks: List[str]) -> str:

    if not co:
        return "Chatbot service is temporarily unavailable."

    context_text = "\n\n".join(context_chunks) if context_chunks else ""

    prompt = f"""
Medical Context:
{context_text}

Patient Question:
{query}

Respond warmly as Dr. Derm.
"""

    try:
        resp = co.chat(
            model=COHERE_MODEL,
            message=prompt,
            preamble=SYSTEM_PROMPT,
            max_tokens=500,
            temperature=0.4,
        )
        return (resp.text or "").strip()

    except Exception as e:
        print("Cohere error:", e)
        return "I’m sorry, I’m having trouble responding right now."


# ─────────────────────────────────────────────
# 9. API Models
# ─────────────────────────────────────────────
class ChatIn(BaseModel):
    message: str


# ─────────────────────────────────────────────
# 10. Routes
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home():
    return "<h2>DermAI Chatbot Running</h2>"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks": len(STATE["original_docs"]),
        "cohere_enabled": bool(co),
    }


@app.post("/chat")
def chat(payload: ChatIn):

    q = payload.message.strip()

    if not q:
        return {"ok": False, "error": "Empty message."}

    try:
        ctx = find_best_matches(q)
        ans = derma_answer(q, ctx)

        return {
            "ok": True,
            "answer": ans
        }

    except Exception as e:
        print("Chat error:", e)
        return {"ok": False, "error": "Chat failed."}
