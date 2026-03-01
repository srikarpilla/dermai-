"""
backend.py — DermAI Integrated Server
Runs Flask (skin prediction) + mounts FastAPI (RAG chatbot) together
on a single port using DispatcherMiddleware.
Works on Render, HuggingFace, and locally.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

import tensorflow as tf
import numpy as np
import json
from PIL import Image
import io
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

# ─────────────────────────────────────────────
#  Paths — relative (works on all platforms)
# ─────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH     = os.path.join(BASE_DIR, "best_weights.weights.h5")
ARCH_PATH        = os.path.join(BASE_DIR, "model_architecture.json")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
SYMPTOMS_PATH    = os.path.join(BASE_DIR, "symptoms.json")
MEDICINES_PATH   = os.path.join(BASE_DIR, "medicines.json")
IMG_SIZE         = (224, 224)

# ─────────────────────────────────────────────
#  Email — from environment variables
#  Set these in Render Dashboard → Environment
# ─────────────────────────────────────────────
SENDER_EMAIL    = os.environ.get("SENDER_EMAIL", "")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD", "")

# ─────────────────────────────────────────────
#  Flask App
# ─────────────────────────────────────────────
flask_app = Flask(__name__, static_folder=BASE_DIR, static_url_path='')
CORS(flask_app)

# ─────────────────────────────────────────────
#  Load class names
# ─────────────────────────────────────────────
print("Loading class names...")
with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
    class_names = json.load(f)
print(f"{len(class_names)} classes loaded.")

# ─────────────────────────────────────────────
#  Load model — using model_architecture.json
#  + best_weights.weights.h5 (your original files)
# ─────────────────────────────────────────────
print("Loading model architecture from JSON...")
with open(ARCH_PATH, 'r', encoding='utf-8') as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)
print("Loading trained weights...")
model.load_weights(WEIGHTS_PATH)
print("Model loaded successfully!")

# ─────────────────────────────────────────────
#  Load symptoms and medicines
# ─────────────────────────────────────────────
with open(SYMPTOMS_PATH, 'r', encoding='utf-8') as f:
    DISEASE_SYMPTOMS = json.load(f)

with open(MEDICINES_PATH, 'r', encoding='utf-8') as f:
    MEDICINES_DB = json.load(f)

# ─────────────────────────────────────────────
#  Warm up model (prevents first-request timeout)
# ─────────────────────────────────────────────
print("Warming up model...")
_dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
model.predict(_dummy, verbose=0)
print("Model warmed up. Server is ready!")


# ─────────────────────────────────────────────
#  Image Preprocessing
# ─────────────────────────────────────────────
def preprocess_image(img_bytes):
    img       = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = (img_array / 127.5) - 1.0
    return img_array


def get_medicine_info(disease_name):
    return MEDICINES_DB.get(disease_name, {})


# ─────────────────────────────────────────────
#  Medicine HTML Block (for email)
# ─────────────────────────────────────────────
def build_medicine_html_block(meds):
    if not meds:
        return "<p style='color:#6b7280;'>No specific medication data available.</p>"

    SECTION_LABELS = {
        "topical": "Topical Treatment", "oral_moderate": "Oral Treatment (Moderate)",
        "oral_severe": "Oral Treatment (Severe)", "systemic": "Systemic Treatment",
        "first_line": "First-Line Treatment", "adjuvants": "Adjuvant Therapy",
        "antipruritic": "Antipruritic (Itch Relief)", "antihistamines": "Antihistamines",
        "emollients": "Emollients / Moisturizers", "surgical": "Surgical Treatment",
        "immunotherapy": "Immunotherapy", "targeted_therapy": "Targeted Therapy",
        "biologics_psoriasis": "Biologic Therapy", "systemic_psoriasis": "Systemic (Psoriasis)",
        "topical_psoriasis": "Topical (Psoriasis)", "lichen_planus": "Lichen Planus Treatment",
        "rosacea_specific": "Rosacea-Specific", "vitiligo": "Vitiligo Treatment",
        "melasma": "Melasma Treatment", "photodermatoses": "Photodermatosis Treatment",
        "actinic_keratosis": "Actinic Keratosis", "basal_cell_carcinoma": "Basal Cell Carcinoma",
        "squamous_cell_carcinoma": "Squamous Cell Carcinoma",
        "impetigo_topical": "Impetigo (Topical)", "cellulitis_oral": "Cellulitis (Oral)",
        "severe_iv": "Severe / IV Therapy", "scabies_first_line": "Scabies First-Line",
        "scabies_adjuncts": "Scabies Adjuncts", "lyme_disease": "Lyme Disease",
        "insect_bites": "Insect Bite Relief", "onychomycosis_topical": "Nail Fungus (Topical)",
        "onychomycosis_oral": "Nail Fungus (Oral)", "nail_psoriasis": "Nail Psoriasis",
        "tinea_topical": "Tinea (Topical)", "tinea_oral": "Tinea (Oral)",
        "candidiasis": "Candidiasis", "acute_urticaria": "Acute Urticaria",
        "chronic_urticaria": "Chronic Urticaria", "anaphylaxis_emergency": "Anaphylaxis Emergency",
        "herpes_simplex": "Herpes Simplex", "herpes_zoster": "Herpes Zoster",
        "hpv_warts": "HPV / Warts", "androgenetic_alopecia": "Androgenetic Alopecia",
        "alopecia_areata": "Alopecia Areata", "telogen_effluvium": "Telogen Effluvium",
        "tinea_capitis": "Tinea Capitis", "mild": "Mild Cases",
        "moderate_to_severe": "Moderate-Severe Cases",
        "allergic_contact_dermatitis": "Allergic Contact Dermatitis",
        "mild_to_moderate": "Mild to Moderate", "sjs_ten_emergency": "SJS / TEN Emergency",
        "infantile_hemangioma": "Infantile Hemangioma", "pyogenic_granuloma": "Pyogenic Granuloma",
        "port_wine_stain": "Port Wine Stain", "cherry_angioma": "Cherry Angioma",
        "cutaneous_small_vessel": "Cutaneous Vasculitis", "systemic_vasculitis": "Systemic Vasculitis",
        "cutaneous_lupus": "Cutaneous Lupus", "systemic_lupus": "Systemic Lupus (SLE)",
        "warts": "Warts Treatment", "molluscum_contagiosum": "Molluscum Contagiosum",
        "viral_skin_infections_general": "General Viral Care",
        "seborrheic_keratosis": "Seborrheic Keratosis", "dermatofibroma": "Dermatofibroma",
        "lipoma": "Lipoma", "general_approach": "General Approach",
        "diabetes_related": "Diabetes-Related Skin", "thyroid_related": "Thyroid-Related Skin",
        "liver_disease": "Liver Disease Skin", "bullous_pemphigoid": "Bullous Pemphigoid",
        "wound_care": "Wound / Erosion Care", "topical_steroids": "Topical Steroids",
        "calcineurin_inhibitors": "Calcineurin Inhibitors", "supportive": "Supportive Care",
    }
    SKIP_KEYS = {"monitoring", "caution"}
    html = ""
    for key, value in meds.items():
        if key in SKIP_KEYS:
            continue
        label = SECTION_LABELS.get(key, key.replace("_", " ").title())
        if isinstance(value, list):
            items = "".join(f"<li style='margin-bottom:4px;'>{item}</li>" for item in value)
            html += f"""
            <div style="margin-bottom:16px;padding:12px 16px;background:#f9fafb;
                        border-radius:6px;border-left:3px solid #0d9488;">
              <div style="font-size:12px;font-weight:700;color:#0d9488;
                          text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">💊 {label}</div>
              <ul style="margin:0;padding-left:18px;color:#374151;font-size:14px;line-height:1.8;">{items}</ul>
            </div>"""
        elif isinstance(value, str):
            html += f"""
            <div style="margin-bottom:12px;padding:10px 16px;background:#f9fafb;
                        border-radius:6px;border-left:3px solid #0d9488;">
              <div style="font-size:12px;font-weight:700;color:#0d9488;
                          text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">📌 {label}</div>
              <p style="margin:0;color:#374151;font-size:14px;">{value}</p>
            </div>"""
    if meds.get("monitoring"):
        html += f"""<div style="background:#f0f9ff;border-left:4px solid #0ea5e9;
                    padding:12px 16px;border-radius:4px;margin-top:10px;font-size:13px;color:#0c4a6e;">
          <strong>📊 Monitoring:</strong> {meds['monitoring']}</div>"""
    if meds.get("caution"):
        html += f"""<div style="background:#fff7ed;border-left:4px solid #f59e0b;
                    padding:12px 16px;border-radius:4px;margin-top:10px;font-size:13px;color:#92400e;">
          <strong>⚠️ Caution:</strong> {meds['caution']}</div>"""
    return html


def build_medicine_plain_block(meds):
    if not meds:
        return "  No specific medication data available.\n"
    SKIP_KEYS = {"monitoring", "caution"}
    text = ""
    for key, value in meds.items():
        if key in SKIP_KEYS:
            continue
        label = key.replace("_", " ").upper()
        if isinstance(value, list):
            text += f"\n  [{label}]\n" + "".join(f"    - {item}\n" for item in value)
        elif isinstance(value, str):
            text += f"\n  [{label}]\n    {value}\n"
    if meds.get("monitoring"):
        text += f"\n  [MONITORING]\n    {meds['monitoring']}\n"
    if meds.get("caution"):
        text += f"\n  [CAUTION]\n    {meds['caution']}\n"
    return text


# ─────────────────────────────────────────────
#  Email HTML Builder
# ─────────────────────────────────────────────
def build_email_html(name, age, email, phone, symptoms_text,
                     disease, confidence, match_score, matching, missing, meds):
    report_date  = datetime.now().strftime("%B %d, %Y  %H:%M")
    matching_str = ", ".join(matching) if matching else "None"
    missing_str  = ", ".join(missing)  if missing  else "None"
    med_block    = build_medicine_html_block(meds)
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<style>
body{{font-family:'Segoe UI',Arial,sans-serif;background:#f4f4f4;margin:0;padding:0;}}
.wrapper{{max-width:700px;margin:30px auto;background:#fff;border-radius:10px;box-shadow:0 4px 18px rgba(0,0,0,0.10);overflow:hidden;}}
.header{{background:#0d9488;padding:28px 36px;color:#fff;}}
.header h1{{font-size:24px;margin:0 0 4px;letter-spacing:1px;}}
.header p{{margin:0;font-size:13px;opacity:0.85;}}
.info-bar{{background:#f0fdfa;padding:16px 36px;border-bottom:1px solid #ccf0ec;display:flex;gap:40px;flex-wrap:wrap;}}
.info-bar span{{font-size:13px;color:#374151;}}
.info-bar strong{{color:#0d9488;}}
.body{{padding:28px 36px;}}
.sec{{font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:1.2px;color:#0d9488;margin:24px 0 12px;border-bottom:1px solid #e5f4f3;padding-bottom:4px;}}
table{{width:100%;border-collapse:collapse;font-size:15px;}}
td{{padding:9px 6px;vertical-align:top;}}
td.lbl{{width:45%;color:#6b7280;font-weight:500;}}
td.val{{color:#111827;font-weight:600;}}
tr:nth-child(even) td{{background:#f9fafb;}}
.disclaimer{{background:#fff7ed;border-left:4px solid #f59e0b;padding:12px 16px;border-radius:4px;font-size:13px;color:#92400e;margin-top:20px;}}
.footer{{background:#f0fdfa;padding:18px 36px;font-size:12px;color:#6b7280;border-top:1px solid #ccf0ec;text-align:center;}}
</style></head><body>
<div class="wrapper">
  <div class="header"><h1>🩺 DermAI — Skin Condition Report</h1><p>Generated on {report_date}</p></div>
  <div class="info-bar">
    <span><strong>Patient:</strong> {name}</span>
    <span><strong>Age:</strong> {age}</span>
    <span><strong>Email:</strong> {email}</span>
    {"<span><strong>Phone:</strong> " + phone + "</span>" if phone else ""}
  </div>
  <div class="body">
    <p style="font-size:15px;color:#374151;margin-top:0;">Dear <strong>{name}</strong>,<br><br>
    Thank you for using <strong>DermAI</strong>. Below is your analysis report.</p>
    <div class="sec">📋 Analysis Summary</div>
    <table>
      <tr><td class="lbl">Predicted Condition</td><td class="val">{disease}</td></tr>
      <tr><td class="lbl">Confidence Level</td><td class="val">{confidence}%</td></tr>
      <tr><td class="lbl">Symptom Alignment</td><td class="val">{match_score}</td></tr>
      <tr><td class="lbl">Symptoms Reported</td><td class="val">{symptoms_text or "None"}</td></tr>
      <tr><td class="lbl">Matching Symptoms</td><td class="val">{matching_str}</td></tr>
      <tr><td class="lbl">Additional Notes</td><td class="val" style="color:#6b7280;font-weight:400;">{missing_str}</td></tr>
    </table>
    <div class="sec">💊 Medications & Treatment</div>
    {med_block}
    <div class="sec">🛡️ General Skin Care Tips</div>
    <ul style="padding-left:20px;color:#374151;font-size:14px;line-height:2.0;">
      <li>Keep the affected area clean and dry at all times.</li>
      <li>Avoid scratching, rubbing, or picking at the skin.</li>
      <li>Use gentle, fragrance-free moisturizers if dryness is present.</li>
      <li>Apply broad-spectrum SPF 30+ sunscreen every morning.</li>
      <li>Consult a licensed dermatologist for a confirmed diagnosis.</li>
    </ul>
    <div class="disclaimer">⚠️ <strong>Disclaimer:</strong> For informational purposes only.
    Do <strong>NOT</strong> self-medicate. Always consult a qualified healthcare professional.</div>
  </div>
  <div class="footer"><strong>DermAI</strong> — For educational use only<br>
  This is an automated report. Please do not reply to this email.</div>
</div></body></html>"""


def build_email_plain(name, age, symptoms_text,
                      disease, confidence, match_score, matching, missing, meds):
    report_date  = datetime.now().strftime("%B %d, %Y %H:%M")
    matching_str = ", ".join(matching) if matching else "None"
    missing_str  = ", ".join(missing)  if missing  else "None"
    med_text     = build_medicine_plain_block(meds)
    return f"""
============================================================
        DermAI — Skin Condition Report
        Generated: {report_date}
============================================================
Dear {name},

ANALYSIS SUMMARY
  Predicted Condition : {disease}
  Confidence Level    : {confidence}%
  Symptom Alignment   : {match_score}
  Symptoms Reported   : {symptoms_text or "None"}
  Matching Symptoms   : {matching_str}
  Additional Notes    : {missing_str}

RECOMMENDED MEDICATIONS
{med_text}

GENERAL TIPS
  - Keep skin clean and dry.
  - Avoid scratching.
  - Use fragrance-free moisturizers.
  - Apply SPF 30+ sunscreen daily.
  - Consult a dermatologist.

------------------------------------------------------------
DISCLAIMER: For informational purposes only.
Do NOT self-medicate. Consult a qualified healthcare professional.
------------------------------------------------------------
DermAI — For educational use only
"""


# ─────────────────────────────────────────────
#  Send Email
# ─────────────────────────────────────────────
def send_report_email(recipient_email, recipient_name, age, phone, symptoms_text,
                      disease, confidence, match_score, matching, missing, meds):
    try:
        if not SENDER_EMAIL or not SENDER_PASSWORD:
            print("Email credentials not set. Skipping email.")
            return False
        msg            = MIMEMultipart("alternative")
        msg["Subject"] = f"DermAI Report — {disease}"
        msg["From"]    = f"DermAI <{SENDER_EMAIL}>"
        msg["To"]      = recipient_email
        plain = build_email_plain(recipient_name, age, symptoms_text,
                                  disease, confidence, match_score, matching, missing, meds)
        html  = build_email_html(recipient_name, age, recipient_email, phone, symptoms_text,
                                 disease, confidence, match_score, matching, missing, meds)
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(html,  "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())
        print(f"Email sent to {recipient_email}")
        return True
    except Exception as e:
        print(f"Email failed: {e}")
        return False


# ─────────────────────────────────────────────
#  Flask Routes
# ─────────────────────────────────────────────
@flask_app.route('/')
def serve_index():
    return send_from_directory(BASE_DIR, 'index.html')

@flask_app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(BASE_DIR, filename)

@flask_app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        user_info = {}
        if 'user_info' in request.form:
            try:
                user_info = json.loads(request.form['user_info'])
            except Exception:
                pass

        user_name     = user_info.get("name", "Patient")
        user_age      = user_info.get("age", "N/A")
        user_email    = user_info.get("email", "")
        user_phone    = user_info.get("phone", "")
        symptoms_text = user_info.get("symptoms", "")
        user_symptoms = [s.strip() for s in symptoms_text.replace(",", " ").split() if s.strip()]

        img_bytes     = file.read()
        processed_img = preprocess_image(img_bytes)
        predictions   = model.predict(processed_img, verbose=0)[0]

        top_idx           = int(np.argmax(predictions))
        confidence        = float(predictions[top_idx] * 100)
        predicted_disease = class_names[top_idx]

        known_symptoms = DISEASE_SYMPTOMS.get(predicted_disease, [])
        matching = [s for s in user_symptoms if any(s.lower() == k.lower() for k in known_symptoms)]
        missing  = [k for k in known_symptoms if not any(k.lower() == u.lower() for u in user_symptoms)]
        match_score = (
            f"{len(matching)} of {len(known_symptoms)} typical symptoms match"
            if known_symptoms else "No symptom data available"
        )
        confidence_str = f"{confidence:.2f}"
        meds = get_medicine_info(predicted_disease)

        email_sent = False
        if user_email:
            email_sent = send_report_email(
                recipient_email=user_email, recipient_name=user_name,
                age=user_age, phone=user_phone, symptoms_text=symptoms_text,
                disease=predicted_disease, confidence=confidence_str,
                match_score=match_score, matching=matching, missing=missing, meds=meds
            )

        return jsonify({
            "disease": predicted_disease, "confidence": confidence_str,
            "match_score": match_score, "matching": matching,
            "missing": missing, "medicines": meds, "email_sent": email_sent
        })
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": "Prediction failed"}), 500


# ─────────────────────────────────────────────
#  Import and mount FastAPI chatbot
# ─────────────────────────────────────────────
from derma_chat import app as fastapi_app
from a2wsgi import ASGIMiddleware

# Mount FastAPI at /chat-ui prefix
combined_app = DispatcherMiddleware(flask_app, {
    '/chat-ui': ASGIMiddleware(fastapi_app),
})


# ─────────────────────────────────────────────
#  Run
# ─────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"\nDermAI integrated server starting on port {port}\n")
    run_simple('0.0.0.0', port, combined_app, use_reloader=False, use_debugger=False)
