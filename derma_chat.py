"""
derma_chat.py — DermAI RAG Chatbot (FastAPI)
Mounted at /chat-ui by backend.py via DispatcherMiddleware.
Do NOT run this directly — it is imported and mounted by backend.py.
"""

import os
import ssl
import json
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
COHERE_MODEL   = os.environ.get("COHERE_MODEL", "command-r-08-2024").strip()

if not COHERE_API_KEY:
    raise RuntimeError("COHERE_API_KEY not set. Add it to Render Environment Variables.")

# ─────────────────────────────────────────────
# 2. NLTK setup
# ─────────────────────────────────────────────
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

ps         = PorterStemmer()
CHUNK_SIZE = 1000
TOP_N      = 4

# ─────────────────────────────────────────────
# 3. FastAPI app
# ─────────────────────────────────────────────
app = FastAPI(title="DermAI Dermatologist Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

co = cohere.Client(api_key=COHERE_API_KEY)

# ─────────────────────────────────────────────
# 4. In-memory vector store
# ─────────────────────────────────────────────
STATE = {
    "original_docs":  [],
    "processed_docs": [],
    "vectorizer":     TfidfVectorizer(),
    "vectors":        None,
}

# ─────────────────────────────────────────────
# 5. Built-in dermatology knowledge base
# ─────────────────────────────────────────────
DERMA_KNOWLEDGE = """
ACNE AND ROSACEA
Acne is a chronic skin condition that occurs when hair follicles become plugged with oil and dead skin cells. It is caused by excess sebum production, Cutibacterium acnes bacteria, hormonal changes, and inflammation. Acne appears as whiteheads, blackheads, papules, pustules, nodules, or cystic lesions, typically on the face, chest, and back. Triggers include hormonal fluctuations, stress, high-glycemic diets, dairy products, and certain cosmetics. Treatment includes topical benzoyl peroxide, retinoids like tretinoin, salicylic acid, topical or oral antibiotics, and oral isotretinoin for severe cases. Foods to avoid include refined sugar, white bread, processed foods, and high-fat dairy. Helpful foods include omega-3 rich foods like salmon, zinc-rich foods like pumpkin seeds, antioxidant-rich berries, and green leafy vegetables. Precautions: wash face twice daily with gentle cleanser, avoid touching your face, change pillowcases frequently, stay hydrated, and never pop or squeeze pimples. Rosacea is a chronic inflammatory condition causing facial redness, visible blood vessels, and acne-like bumps. It is triggered by sun exposure, heat, alcohol, spicy food, and stress. Treatment includes topical metronidazole, azelaic acid, and oral doxycycline.

ECZEMA AND ATOPIC DERMATITIS
Eczema is a group of inflammatory skin conditions causing itchy, inflamed, and irritated skin. Atopic dermatitis is the most common form, linked to immune system overactivation, genetics, and environmental triggers. It commonly begins in childhood and often coexists with asthma and allergic rhinitis. The skin barrier is defective, losing moisture and allowing allergens and bacteria to penetrate. Triggers include soaps, detergents, dust mites, pet dander, mold, pollen, stress, sweating, and synthetic fabrics. Symptoms include intense itching especially at night, dry scaly patches, red or brownish-grey patches, and small raised bumps. Treatment includes liberal use of emollients and moisturizers immediately after bathing, topical corticosteroids during flares, calcineurin inhibitors like tacrolimus for sensitive areas, and dupilumab biologic for severe cases. Foods that may worsen eczema include dairy, eggs, wheat, soy, nuts, and fish in some patients. Anti-inflammatory foods like fatty fish, turmeric, probiotic yogurt, and colorful vegetables may help. Precautions: take short lukewarm showers, use fragrance-free products, wear soft cotton clothing, keep nails short to minimize scratching damage, and maintain cool indoor temperatures.

PSORIASIS
Psoriasis is a chronic autoimmune condition where the immune system attacks healthy skin cells, speeding up the skin cell life cycle. Cells build up rapidly on the skin surface forming scales and red patches. It is not contagious. Types include plaque psoriasis, guttate psoriasis, inverse psoriasis, pustular psoriasis, and erythrodermic psoriasis. Triggers include infections, stress, smoking, heavy alcohol consumption, certain medications like lithium and beta-blockers, and skin injuries. Treatment includes topical corticosteroids, vitamin D analogues like calcipotriol, coal tar, methotrexate, cyclosporine, acitretin, and biologics like adalimumab, secukinumab, and ixekizumab. Anti-inflammatory diet rich in omega-3 fatty acids, fruits, vegetables, and whole grains can help. Foods to avoid include red meat, processed foods, refined sugar, alcohol, and gluten in some patients. Precautions: moisturize regularly, avoid known triggers, quit smoking, limit alcohol, protect skin from injury, and manage stress through yoga and meditation.

FUNGAL INFECTIONS
Dermatophytosis or ringworm is caused by dermatophyte fungi that infect the skin, hair, and nails. Types include tinea corporis (body), tinea capitis (scalp), tinea pedis (athlete's foot), tinea cruris (jock itch), and tinea unguium (nail). Candidiasis is caused by Candida yeast, commonly affecting warm moist areas. These infections spread through direct contact with infected people, animals, or contaminated surfaces. Symptoms include ring-shaped scaly patches, itching, redness, and nail discoloration. Treatment includes topical antifungals like clotrimazole and terbinafine for mild cases, and oral itraconazole or terbinafine for extensive or nail infections. Probiotic-rich foods like yogurt can support immune defense. Avoid sugary foods as sugar feeds yeast growth. Precautions: keep skin dry especially in skin folds, wear breathable cotton clothing, avoid sharing towels or shoes, wear footwear in public showers, and complete full course of antifungal treatment.

CONTACT DERMATITIS
Contact dermatitis is skin inflammation caused by direct contact with an irritant or allergen. Irritant contact dermatitis is caused by soaps, detergents, cleaning products, chemicals, and excessive handwashing. Allergic contact dermatitis is an immune reaction to allergens like nickel in jewelry, latex, hair dyes, cosmetic fragrances, and poison ivy. Symptoms include redness, itching, burning, swelling, blisters, and skin thickening. Treatment includes identifying and avoiding the trigger, topical corticosteroids for inflammation, calamine lotion for itch relief, and oral antihistamines. Precautions: wear protective gloves when handling chemicals, choose hypoallergenic products, do patch testing before using new cosmetics, and moisturize regularly to maintain the skin barrier.

URTICARIA (HIVES)
Urticaria presents as raised, itchy welts on the skin that appear suddenly due to histamine release. Triggers include food allergens like nuts, shellfish, eggs, and strawberries, medications like aspirin and NSAIDs, insect stings, infections, stress, heat, cold, and pressure. Chronic urticaria lasting more than 6 weeks is often autoimmune. Treatment includes non-sedating antihistamines like cetirizine and fexofenadine as first line, and omalizumab biologic for refractory cases. Anaphylaxis is a life-threatening emergency requiring immediate epinephrine injection. Precautions: identify and avoid triggers, carry an epinephrine auto-injector if prescribed, and keep a symptom diary to identify patterns.

BACTERIAL SKIN INFECTIONS
Impetigo is a highly contagious bacterial infection caused by Staphylococcus aureus or Streptococcus pyogenes. It presents as honey-colored crusted sores, typically around the nose and mouth in children. Cellulitis is a bacterial infection of the deeper skin layers causing warmth, redness, swelling, and pain, usually affecting the legs. It requires prompt oral antibiotics and may need IV treatment if severe. Folliculitis is inflammation of hair follicles caused by bacterial or fungal infection. Treatment includes topical mupirocin for impetigo, oral cephalexin or amoxicillin-clavulanate for cellulitis, and antiseptic washes. Precautions: maintain good hygiene, avoid sharing personal items, keep wounds clean and covered, and complete the full antibiotic course.

HERPES AND VIRAL INFECTIONS
Herpes simplex virus causes cold sores around the mouth and genital herpes. The virus remains dormant in nerve cells and reactivates due to stress, illness, sun exposure, or hormonal changes. Herpes zoster or shingles is caused by reactivation of the varicella-zoster virus causing painful blistering along a nerve distribution. Warts are caused by human papillomavirus HPV and are highly contagious. Treatment for herpes includes oral acyclovir, valacyclovir, or famciclovir. Postherpetic neuralgia after shingles is treated with pregabalin or gabapentin. Precautions: avoid close contact during active outbreaks, use barrier protection, do not share utensils or towels, and get vaccinated against varicella and HPV.

MELANOMA AND SKIN CANCER
Melanoma is the most dangerous skin cancer arising from melanocytes. Risk factors include UV radiation from sun and tanning beds, fair skin, family history, many moles, and immunosuppression. The ABCDE criteria for evaluating moles are Asymmetry, Border irregularity, Color variation, Diameter greater than 6mm, and Evolution or change over time. Basal cell carcinoma is the most common skin cancer, growing slowly and rarely spreading. Squamous cell carcinoma arises from sun-damaged skin cells. Treatment for melanoma includes surgical excision, immunotherapy with pembrolizumab or nivolumab, and targeted therapy with BRAF inhibitors. Precautions: wear SPF 30 or higher sunscreen daily, avoid peak sun hours, wear protective clothing, avoid tanning beds, and perform monthly self skin examinations.

SCABIES
Scabies is caused by the mite Sarcoptes scabiei burrowing into the skin and causing intense itching, especially at night. It spreads through prolonged skin-to-skin contact. Symptoms include pimple-like rashes, blisters, and burrow tracks, commonly in finger webs, wrists, waist, genitals, and buttocks. Treatment requires permethrin 5% cream applied from neck to toes. Oral ivermectin is an alternative. All household members must be treated simultaneously. Clothing and bedding must be washed at 60 degrees Celsius.

GENERAL SKIN HEALTH
Healthy skin requires a balanced skincare routine. Cleanse twice daily with a gentle pH-balanced cleanser. Moisturize while skin is still slightly damp to lock in hydration. Use broad-spectrum SPF 30 or higher sunscreen every morning regardless of weather. Drink at least 8 glasses of water daily for internal hydration. Eat a diet rich in antioxidants from colorful fruits and vegetables, omega-3 fatty acids from fatty fish and walnuts, vitamin C from citrus fruits and bell peppers, and zinc from pumpkin seeds and legumes. Avoid smoking, excessive alcohol, and high-sugar diets that accelerate skin aging. Get 7 to 9 hours of quality sleep as skin repairs itself at night. Exercise regularly to improve circulation and nutrient delivery to the skin. Manage stress through meditation, yoga, or mindfulness as chronic stress worsens many skin conditions.

HAIR AND SCALP CONDITIONS
Dandruff is caused by Malassezia yeast overgrowth and is treated with antifungal shampoos containing ketoconazole or selenium sulfide. Seborrheic dermatitis causes oily, flaky patches on the scalp and face and is treated with medicated shampoos and topical antifungals. Alopecia areata is an autoimmune condition causing patchy hair loss, treated with intralesional corticosteroids or topical minoxidil. Androgenetic alopecia is hereditary hair thinning treated with minoxidil, finasteride, or hair transplantation. Telogen effluvium is temporary hair shedding caused by stress, nutritional deficiencies, hormonal changes, or illness. Iron, biotin, and protein deficiency can cause diffuse hair loss.

NAIL CONDITIONS
Onychomycosis is fungal nail infection causing thickening, discoloration, and brittleness. It is treated with oral terbinafine or itraconazole for at least 3 months. Nail psoriasis causes pitting, onycholysis, and subungual hyperkeratosis. Paronychia is infection of the nail fold caused by bacteria or Candida. Clubbing of nails may indicate heart or lung disease. Beau lines are horizontal ridges indicating previous illness or nutritional deficiency.

PHOTODERMATOLOGY
Polymorphic light eruption is the most common sun-induced rash causing itchy bumps after sun exposure. Solar urticaria is an allergic reaction to sunlight causing hives. Phototoxic reactions are caused by medications or plants that sensitize skin to UV light. Photoallergic reactions involve immune-mediated responses to UV-activated allergens. Sunscreens, protective clothing, and avoiding peak sun hours are essential preventive measures for all photodermatoses.
"""


# ─────────────────────────────────────────────
# 6. Build knowledge base at startup
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
    STATE["original_docs"]  = chunks
    STATE["processed_docs"] = [stem_text(c) for c in chunks]
    STATE["vectorizer"]     = TfidfVectorizer()
    STATE["vectors"]        = STATE["vectorizer"].fit_transform(STATE["processed_docs"])
    print(f"Knowledge base built: {len(chunks)} chunks loaded.")


build_knowledge_base()


def find_best_matches(query: str, top_n: int = TOP_N) -> List[str]:
    if STATE["vectors"] is None or len(STATE["original_docs"]) == 0:
        return []
    q_vec  = STATE["vectorizer"].transform([stem_text(query)])
    scores = (STATE["vectors"] * q_vec.T).toarray().flatten()
    top_idx = scores.argsort()[-top_n:][::-1]
    return [STATE["original_docs"][i] for i in top_idx if scores[i] > 0]


# ─────────────────────────────────────────────
# 7. Cohere answer generation
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are Dr. Derm, a warm, knowledgeable, and empathetic AI dermatologist.
You speak like a real doctor talking to a patient — clear, caring, and human.
You explain things in plain language, never use cold robotic phrases, and always show concern for the patient's wellbeing.

Guidelines:
- Greet the patient naturally when appropriate
- Use a conversational, warm tone — like a doctor sitting across from a patient
- Give detailed, practical answers covering: what the condition is, how it develops, causes, food and lifestyle advice, medications, and precautions
- Break your answer into easy-to-read sections using simple formatting
- Always recommend consulting a dermatologist for diagnosis and prescription
- If you don't have enough context, be honest but still provide general guidance
- Never give a definitive diagnosis — always say "this could be" or "sounds like it might be"
- End responses with a caring closing line or offer to answer follow-up questions
- Use "I" naturally in your responses as a doctor would"""


def derma_answer(query: str, context_chunks: List[str]) -> str:
    if context_chunks:
        context_text = "\n\n".join(
            [f"[Medical Reference {i+1}]\n{c}" for i, c in enumerate(context_chunks)]
        )
        prompt = (
            f"Medical Reference Context:\n{context_text}\n\n"
            f"Patient's Question: {query}\n\n"
            f"Respond as Dr. Derm — warm, detailed, and human."
        )
    else:
        prompt = (
            f"Patient's Question: {query}\n\n"
            f"Respond as Dr. Derm — warm, detailed, and human."
        )
    resp = co.chat(
        model=COHERE_MODEL,
        message=prompt,
        preamble=SYSTEM_PROMPT,
        max_tokens=600,
        temperature=0.4,
    )
    return (resp.text or "").strip()


# ─────────────────────────────────────────────
# 8. API Models
# ─────────────────────────────────────────────
class ChatIn(BaseModel):
    message: str

class SuggestIn(BaseModel):
    disease: str


# ─────────────────────────────────────────────
# 9. Routes
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home():
    html_path = os.path.join(os.path.dirname(__file__), "derma_chat.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>DermAI Chat</h1><p>derma_chat.html not found.</p>"


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(STATE["original_docs"]), "model": COHERE_MODEL}


@app.post("/chat")
def chat(payload: ChatIn):
    q = payload.message.strip()
    if not q:
        return {"ok": False, "error": "Empty message."}
    try:
        ctx = find_best_matches(q)
        ans = derma_answer(q, ctx)
        return {"ok": True, "answer": ans}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/suggest")
def suggest_questions(payload: SuggestIn):
    disease = payload.disease.strip()
    questions = [
        f"What exactly is {disease} and how does it develop?",
        f"What are the main causes of {disease}?",
        f"What foods should I eat or avoid if I have {disease}?",
        f"What medicines are commonly used to treat {disease}?",
        f"What daily precautions should I take for {disease}?",
        f"Can {disease} be cured completely?",
        f"Is {disease} contagious?",
        f"What lifestyle changes help with {disease}?",
    ]
    return {"ok": True, "questions": questions}
    
