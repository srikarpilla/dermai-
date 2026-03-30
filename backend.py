"""
DermAI Backend — Flask API
  • Standard SMTP email (Gmail App Password)
  • Two-email flow:
      1. Patient gets an immediate AI report + "awaiting doctor verification" notice
      2. Doctor (srikarpilla2@gmail.com) gets the full case for review + approve link
  • Doctor Portal  (/doctor)  — token-authenticated one-click prescription approval
"""

from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image
import io
import os
import hmac
import hashlib
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder='.', static_url_path='')

# ─────────────────────────────────────────────────────────────────────────────
#  Config  (set as environment variables or in a .env file)
# ─────────────────────────────────────────────────────────────────────────────
WEIGHTS_PATH     = os.getenv("WEIGHTS_PATH",     "best_weights.weights.h5")
ARCH_PATH        = os.getenv("ARCH_PATH",         "model_architecture.json")
CLASS_NAMES_PATH = os.getenv("CLASS_NAMES_PATH",  "class_names.json")
SYMPTOMS_PATH    = os.getenv("SYMPTOMS_PATH",     "symptoms.json")
MEDICINES_PATH   = os.getenv("MEDICINES_PATH",    "medicines.json")
IMG_SIZE         = (224, 224)
PORT             = int(os.getenv("PORT", 5000))

# ── SMTP config ───────────────────────────────────────────────────────────────
# For Gmail: enable 2FA → generate an App Password at
#   https://myaccount.google.com/apppasswords
# Then add these to your .env file:
#   SMTP_HOST=smtp.gmail.com
#   SMTP_PORT=587
#   SMTP_USER=your.gmail@gmail.com
#   SMTP_PASS=xxxx xxxx xxxx xxxx   ← 16-char App Password
#   SENDER_FROM=DermAI <your.gmail@gmail.com>
SMTP_HOST   = os.getenv("SMTP_HOST",   "smtp.gmail.com")
SMTP_PORT   = int(os.getenv("SMTP_PORT", 587))
SMTP_USER   = os.getenv("SMTP_USER",   "")   # your Gmail address
SMTP_PASS   = os.getenv("SMTP_PASS",   "")   # Gmail App Password (16 chars, no spaces)
SENDER_FROM = os.getenv("SENDER_FROM", f"DermAI <{SMTP_USER}>")

# Doctor portal
DOCTOR_EMAIL = "srikarpilla2@gmail.com"
TOKEN_SECRET = os.getenv("TOKEN_SECRET", "dermai-doctor-secret-2024")
BASE_URL     = os.getenv("BASE_URL",     "http://127.0.0.1:5000")

# ─────────────────────────────────────────────────────────────────────────────
#  Load Model & Data
# ─────────────────────────────────────────────────────────────────────────────
print("Loading model architecture...")
with open(ARCH_PATH, 'r', encoding='utf-8') as f:
    model_json = f.read()
model = tf.keras.models.model_from_json(model_json)

print("Loading weights...")
model.load_weights(WEIGHTS_PATH)

with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
    class_names = json.load(f)

with open(SYMPTOMS_PATH, 'r', encoding='utf-8') as f:
    DISEASE_SYMPTOMS = json.load(f)

with open(MEDICINES_PATH, 'r', encoding='utf-8') as f:
    MEDICINES_DB = json.load(f)

print(f"Model ready — {len(class_names)} classes.")

# ─────────────────────────────────────────────────────────────────────────────
#  In-memory case store  (swap for a DB in production)
# ─────────────────────────────────────────────────────────────────────────────
PENDING_CASES: dict[str, dict] = {}   # token → case_data

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_image(img_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize(IMG_SIZE)
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = (arr / 127.5) - 1.0
    return arr


def get_medicine_info(disease_name: str) -> dict:
    return MEDICINES_DB.get(disease_name, {})


def make_token(patient_email: str, disease: str) -> str:
    """Simple HMAC-based token encoding patient email + disease + timestamp."""
    payload = f"{patient_email}|{disease}|{int(time.time())}"
    sig = hmac.new(TOKEN_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()[:16]
    import base64
    b64 = base64.urlsafe_b64encode(payload.encode()).decode()
    return f"{b64}.{sig}"


def store_case(token: str, case: dict):
    PENDING_CASES[token] = case


def get_case(token: str) -> dict | None:
    return PENDING_CASES.get(token)


# ─────────────────────────────────────────────────────────────────────────────
#  SMTP Email sender
# ─────────────────────────────────────────────────────────────────────────────
def send_via_smtp(to: str | list, subject: str, html: str, text: str) -> bool:
    """Send an email using standard SMTP with TLS (Gmail-compatible)."""
    if not SMTP_USER or not SMTP_PASS:
        print("⚠️  SMTP_USER or SMTP_PASS not set — skipping email")
        return False

    recipients = [to] if isinstance(to, str) else to

    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From']    = SENDER_FROM
        msg['To']      = ", ".join(recipients)

        # Plain text first, HTML second (clients prefer the last part)
        msg.attach(MIMEText(text, 'plain'))
        msg.attach(MIMEText(html, 'html'))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()   # upgrade to TLS
            server.ehlo()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, recipients, msg.as_string())

        print(f"✉️  Email sent to {to}")
        return True

    except smtplib.SMTPAuthenticationError:
        print("❌ SMTP auth failed — check SMTP_USER / SMTP_PASS in your .env")
        return False
    except smtplib.SMTPException as exc:
        print(f"❌ SMTP error: {exc}")
        return False
    except Exception as exc:
        print(f"❌ Email send failed: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  HTML / Text builders — Medicine block (shared)
# ─────────────────────────────────────────────────────────────────────────────
SECTION_LABELS = {
    "topical": "Topical Treatment", "oral_moderate": "Oral (Moderate)",
    "oral_severe": "Oral (Severe)", "systemic": "Systemic Treatment",
    "first_line": "First-Line Treatment", "adjuvants": "Adjuvant Therapy",
    "antipruritic": "Antipruritic", "antihistamines": "Antihistamines",
    "emollients": "Emollients / Moisturizers", "surgical": "Surgical",
    "immunotherapy": "Immunotherapy", "targeted_therapy": "Targeted Therapy",
    "biologics_psoriasis": "Biologic Therapy", "systemic_psoriasis": "Systemic (Psoriasis)",
    "topical_psoriasis": "Topical (Psoriasis)", "lichen_planus": "Lichen Planus",
    "rosacea_specific": "Rosacea-Specific", "vitiligo": "Vitiligo",
    "melasma": "Melasma", "photodermatoses": "Photodermatosis",
    "actinic_keratosis": "Actinic Keratosis", "basal_cell_carcinoma": "Basal Cell Carcinoma",
    "squamous_cell_carcinoma": "Squamous Cell Carcinoma",
    "impetigo_topical": "Impetigo (Topical)", "cellulitis_oral": "Cellulitis (Oral)",
    "severe_iv": "Severe / IV Therapy", "scabies_first_line": "Scabies First-Line",
    "scabies_adjuncts": "Scabies Adjuncts", "lyme_disease": "Lyme Disease",
    "insect_bites": "Insect Bites", "onychomycosis_topical": "Nail Fungus (Topical)",
    "onychomycosis_oral": "Nail Fungus (Oral)", "nail_psoriasis": "Nail Psoriasis",
    "tinea_topical": "Tinea (Topical)", "tinea_oral": "Tinea (Oral)",
    "candidiasis": "Candidiasis", "acute_urticaria": "Acute Urticaria",
    "chronic_urticaria": "Chronic Urticaria", "anaphylaxis_emergency": "Anaphylaxis Emergency",
    "herpes_simplex": "Herpes Simplex", "herpes_zoster": "Herpes Zoster",
    "hpv_warts": "HPV / Warts", "androgenetic_alopecia": "Androgenetic Alopecia",
    "alopecia_areata": "Alopecia Areata", "telogen_effluvium": "Telogen Effluvium",
    "tinea_capitis": "Tinea Capitis", "mild": "Mild Cases",
    "moderate_to_severe": "Moderate-Severe", "mild_to_moderate": "Mild to Moderate",
    "allergic_contact_dermatitis": "Allergic Contact Dermatitis",
    "sjs_ten_emergency": "SJS / TEN Emergency",
    "infantile_hemangioma": "Infantile Hemangioma", "pyogenic_granuloma": "Pyogenic Granuloma",
    "port_wine_stain": "Port Wine Stain", "cherry_angioma": "Cherry Angioma",
    "cutaneous_small_vessel": "Cutaneous Vasculitis", "systemic_vasculitis": "Systemic Vasculitis",
    "cutaneous_lupus": "Cutaneous Lupus", "systemic_lupus": "Systemic Lupus (SLE)",
    "warts": "Warts", "molluscum_contagiosum": "Molluscum Contagiosum",
    "viral_skin_infections_general": "General Viral Care",
    "seborrheic_keratosis": "Seborrheic Keratosis", "dermatofibroma": "Dermatofibroma",
    "lipoma": "Lipoma", "general_approach": "General Approach",
    "diabetes_related": "Diabetes-Related", "thyroid_related": "Thyroid-Related",
    "liver_disease": "Liver Disease", "bullous_pemphigoid": "Bullous Pemphigoid",
    "wound_care": "Wound Care", "topical_steroids": "Topical Steroids",
    "calcineurin_inhibitors": "Calcineurin Inhibitors", "supportive": "Supportive Care",
}
SKIP_KEYS = {"monitoring", "caution"}


def build_medicine_html_block(meds: dict) -> str:
    if not meds:
        return "<p style='color:#6b7280;'>No specific medication data available.</p>"
    html = ""
    for key, value in meds.items():
        if key in SKIP_KEYS:
            continue
        label = SECTION_LABELS.get(key, key.replace("_", " ").title())
        if isinstance(value, list):
            items = "".join(f"<li style='margin-bottom:4px;'>{v}</li>" for v in value)
            html += f"""
            <div style="margin-bottom:16px;padding:12px 16px;background:#f9fafb;
                        border-radius:6px;border-left:3px solid #0d9488;">
              <div style="font-size:12px;font-weight:700;color:#0d9488;
                          text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">
                💊 {label}</div>
              <ul style="margin:0;padding-left:18px;color:#374151;font-size:14px;line-height:1.8;">
                {items}</ul>
            </div>"""
        elif isinstance(value, str):
            html += f"""
            <div style="margin-bottom:12px;padding:10px 16px;background:#f9fafb;
                        border-radius:6px;border-left:3px solid #0d9488;">
              <div style="font-size:12px;font-weight:700;color:#0d9488;
                          text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">
                📌 {label}</div>
              <p style="margin:0;color:#374151;font-size:14px;">{value}</p>
            </div>"""
    if meds.get("monitoring"):
        html += f"""
        <div style="background:#f0f9ff;border-left:4px solid #0ea5e9;padding:12px 16px;
                    border-radius:4px;margin-top:10px;font-size:13px;color:#0c4a6e;">
          <strong>📊 Monitoring:</strong> {meds['monitoring']}</div>"""
    if meds.get("caution"):
        html += f"""
        <div style="background:#fff7ed;border-left:4px solid #f59e0b;padding:12px 16px;
                    border-radius:4px;margin-top:10px;font-size:13px;color:#92400e;">
          <strong>⚠️ Caution:</strong> {meds['caution']}</div>"""
    return html


def build_medicine_plain_block(meds: dict) -> str:
    if not meds:
        return "  No specific medication data available.\n"
    text = ""
    for key, value in meds.items():
        if key in SKIP_KEYS:
            continue
        label = key.replace("_", " ").upper()
        if isinstance(value, list):
            text += f"\n  [{label}]\n" + "".join(f"    - {v}\n" for v in value)
        elif isinstance(value, str):
            text += f"\n  [{label}]\n    {value}\n"
    if meds.get("monitoring"):
        text += f"\n  [MONITORING]\n    {meds['monitoring']}\n"
    if meds.get("caution"):
        text += f"\n  [CAUTION]\n    {meds['caution']}\n"
    return text


# ─────────────────────────────────────────────────────────────────────────────
#  EMAIL 1 — Patient confirmation  (awaiting doctor review)
# ─────────────────────────────────────────────────────────────────────────────
def build_patient_email_html(name, age, email, phone, symptoms_text,
                              disease, confidence, match_score, matching, missing, meds):
    report_date   = datetime.now().strftime("%B %d, %Y  %H:%M")
    matching_str  = ", ".join(matching) if matching else "None"
    missing_str   = ", ".join(missing)  if missing  else "None"
    medicine_html = build_medicine_html_block(meds)

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<style>
  body      {{ font-family:'Segoe UI',Arial,sans-serif; background:#f4f4f4; margin:0; padding:0; }}
  .wrapper  {{ max-width:700px; margin:30px auto; background:#fff; border-radius:10px;
               box-shadow:0 4px 18px rgba(0,0,0,.10); overflow:hidden; }}
  .header   {{ background:#0d9488; padding:28px 36px; color:#fff; }}
  .header h1 {{ font-size:22px; margin:0 0 4px; }}
  .header p  {{ margin:0; font-size:13px; opacity:.85; }}
  .info-bar  {{ background:#f0fdfa; padding:14px 36px; border-bottom:1px solid #ccf0ec; }}
  .info-bar span {{ font-size:13px; color:#374151; margin-right:30px; }}
  .info-bar strong {{ color:#0d9488; }}
  .body      {{ padding:28px 36px; }}
  .sec       {{ font-size:13px; font-weight:700; text-transform:uppercase;
                letter-spacing:1.2px; color:#0d9488; margin:24px 0 12px;
                border-bottom:1px solid #e5f4f3; padding-bottom:4px; }}
  table      {{ width:100%; border-collapse:collapse; font-size:15px; }}
  td         {{ padding:9px 6px; vertical-align:top; }}
  td.lbl     {{ width:45%; color:#6b7280; font-weight:500; }}
  td.val     {{ color:#111827; font-weight:600; }}
  tr:nth-child(even) td {{ background:#f9fafb; }}
  .await-box {{ background:#fffbeb; border:1px solid #fcd34d; border-radius:8px;
                padding:18px 20px; margin:24px 0; }}
  .await-box h3 {{ margin:0 0 8px; color:#92400e; font-size:15px; }}
  .await-box p  {{ margin:0; color:#78350f; font-size:14px; line-height:1.7; }}
  .disclaimer {{ background:#fff7ed; border-left:4px solid #f59e0b; padding:12px 16px;
                 border-radius:4px; font-size:13px; color:#92400e; margin-top:20px; }}
  .footer    {{ background:#f0fdfa; padding:18px 36px; font-size:12px; color:#6b7280;
                border-top:1px solid #ccf0ec; text-align:center; }}
  .footer strong {{ color:#0d9488; }}
</style>
</head>
<body>
<div class="wrapper">
  <div class="header">
    <h1>🩺 DermAI — Skin Condition Analysis Report</h1>
    <p>Generated on {report_date}</p>
  </div>
  <div class="info-bar">
    <span><strong>Patient:</strong> {name}</span>
    <span><strong>Age:</strong> {age}</span>
    <span><strong>Email:</strong> {email}</span>
    {"<span><strong>Phone:</strong> " + phone + "</span>" if phone else ""}
  </div>
  <div class="body">
    <p style="font-size:15px;color:#374151;margin-top:0;">
      Dear <strong>{name}</strong>,<br><br>
      Thank you for using <strong>DermAI Skin Condition Analyzer</strong>.
      Your image has been analyzed and an <em>AI-generated report</em> is below.
    </p>
    <div class="await-box">
      <h3>⏳ Awaiting Doctor Verification</h3>
      <p>
        The AI has predicted a preliminary condition and suggested treatment guidelines.
        <strong>This is NOT a final prescription.</strong><br><br>
        Your case has been forwarded to a qualified dermatologist for review.
        You will receive a <strong>verified prescription email</strong> once the doctor
        has approved — usually within 24 hours.
        Please <strong>do not start any medication</strong> before receiving that confirmation.
      </p>
    </div>
    <div class="sec">📋 AI Analysis Summary</div>
    <table>
      <tr><td class="lbl">Predicted Condition</td><td class="val">{disease}</td></tr>
      <tr><td class="lbl">AI Confidence Level</td><td class="val">{confidence}%</td></tr>
      <tr><td class="lbl">Symptom Alignment</td><td class="val">{match_score}</td></tr>
      <tr><td class="lbl">Symptoms You Reported</td><td class="val">{symptoms_text or "None"}</td></tr>
      <tr><td class="lbl">Matching Symptoms</td><td class="val">{matching_str}</td></tr>
      <tr><td class="lbl">Additional Typical Symptoms</td>
          <td class="val" style="color:#6b7280;font-weight:400;">{missing_str}</td></tr>
    </table>
    <div class="sec">💊 AI-Suggested Treatment Reference (Pending Doctor Approval)</div>
    <p style="font-size:13px;color:#6b7280;margin-top:-8px;margin-bottom:14px;">
      Standard clinical guidelines — <em>reference only</em>. Final medication confirmed by your doctor.
    </p>
    {medicine_html}
    <div class="sec">🛡️ General Skin Care Tips</div>
    <ul style="padding-left:20px;color:#374151;font-size:14px;line-height:2.0;">
      <li>Keep the affected area clean and dry at all times.</li>
      <li>Avoid scratching, rubbing, or picking at the skin.</li>
      <li>Use gentle, fragrance-free moisturizers if dryness is present.</li>
      <li>Apply broad-spectrum SPF 30+ sunscreen every morning.</li>
      <li>Consult a licensed dermatologist for a confirmed diagnosis and treatment.</li>
    </ul>
    <div class="disclaimer">
      ⚠️ <strong>Disclaimer:</strong> This report is for <em>informational purposes only</em>.
      Do <strong>NOT</strong> self-medicate. Wait for the doctor-verified email.
    </div>
  </div>
  <div class="footer">
    <strong>DermAI</strong> — Skin Condition Analyzer &nbsp;|&nbsp; For educational use only<br>
    This is an automated report. Do not reply to this email.
  </div>
</div>
</body>
</html>"""


def build_patient_email_plain(name, age, symptoms_text, disease, confidence,
                               match_score, matching, missing, meds):
    report_date  = datetime.now().strftime("%B %d, %Y %H:%M")
    matching_str = ", ".join(matching) if matching else "None"
    missing_str  = ", ".join(missing)  if missing  else "None"
    med_text     = build_medicine_plain_block(meds)
    return f"""
============================================================
        DermAI — Skin Condition Analysis Report
        Generated: {report_date}
============================================================

Dear {name},

Thank you for using DermAI Skin Condition Analyzer.

⏳ AWAITING DOCTOR VERIFICATION
  Your case has been forwarded to a qualified dermatologist.
  You will receive a VERIFIED PRESCRIPTION EMAIL within 24 hours.
  Please DO NOT start any medication before that confirmation.

PATIENT DETAILS
  Name  : {name}
  Age   : {age}

AI ANALYSIS SUMMARY
  Predicted Condition    : {disease}
  AI Confidence          : {confidence}%
  Symptom Alignment      : {match_score}
  Symptoms You Reported  : {symptoms_text or "None"}
  Matching Symptoms      : {matching_str}
  Additional Notes       : {missing_str}

AI-SUGGESTED TREATMENT REFERENCE (Pending Doctor Approval)
{med_text}

GENERAL SKIN CARE TIPS
  - Keep the affected area clean and dry.
  - Avoid scratching or rubbing the skin.
  - Use gentle, fragrance-free moisturizers if dryness is present.
  - Apply SPF 30+ sunscreen daily.
  - Consult a licensed dermatologist for a confirmed diagnosis.

------------------------------------------------------------
DISCLAIMER: This report is for informational purposes only.
Do NOT self-medicate. Wait for the doctor-verified email.
------------------------------------------------------------

DermAI — Skin Condition Analyzer | For educational use only
"""


# ─────────────────────────────────────────────────────────────────────────────
#  EMAIL 2 — Doctor verification request
# ─────────────────────────────────────────────────────────────────────────────
def build_doctor_email_html(name, age, email, phone, symptoms_text,
                             disease, confidence, match_score, matching, missing,
                             meds, verify_url):
    report_date   = datetime.now().strftime("%B %d, %Y  %H:%M")
    matching_str  = ", ".join(matching) if matching else "None"
    missing_str   = ", ".join(missing)  if missing  else "None"
    medicine_html = build_medicine_html_block(meds)

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<style>
  body      {{ font-family:'Segoe UI',Arial,sans-serif; background:#f1f5f9; margin:0; padding:0; }}
  .wrapper  {{ max-width:720px; margin:30px auto; background:#fff; border-radius:10px;
               box-shadow:0 4px 18px rgba(0,0,0,.12); overflow:hidden; }}
  .header   {{ background:#1e293b; padding:26px 36px; color:#fff; display:flex;
               align-items:center; gap:16px; }}
  .header h1 {{ font-size:20px; margin:0 0 4px; }}
  .header p  {{ margin:0; font-size:13px; color:#94a3b8; }}
  .badge     {{ background:#0d9488; color:#fff; border-radius:6px; padding:6px 14px;
                font-size:12px; font-weight:700; letter-spacing:1px; white-space:nowrap; }}
  .info-bar  {{ background:#f0fdfa; padding:14px 36px; border-bottom:1px solid #ccf0ec; }}
  .info-bar span {{ font-size:13px; color:#374151; margin-right:28px; }}
  .info-bar strong {{ color:#0d9488; }}
  .body      {{ padding:28px 36px; }}
  .sec       {{ font-size:12px; font-weight:700; text-transform:uppercase;
                letter-spacing:1.2px; color:#64748b; margin:24px 0 10px;
                border-bottom:1px solid #e2e8f0; padding-bottom:4px; }}
  table      {{ width:100%; border-collapse:collapse; font-size:14px; }}
  td         {{ padding:8px 6px; vertical-align:top; }}
  td.lbl     {{ width:45%; color:#6b7280; }}
  td.val     {{ color:#0f172a; font-weight:600; }}
  tr:nth-child(even) td {{ background:#f9fafb; }}
  .action-box {{ background:#f0fdf4; border:2px solid #86efac; border-radius:10px;
                 padding:24px 28px; margin:28px 0; text-align:center; }}
  .action-box h3 {{ margin:0 0 10px; color:#166534; font-size:16px; }}
  .action-box p  {{ margin:0 0 18px; color:#15803d; font-size:14px; }}
  .approve-btn   {{ display:inline-block; background:#0d9488; color:#fff;
                    text-decoration:none; border-radius:8px; padding:14px 36px;
                    font-size:15px; font-weight:700; letter-spacing:0.5px;
                    box-shadow:0 4px 14px rgba(13,148,136,0.4); }}
  .url-fallback  {{ font-size:12px; color:#6b7280; margin-top:12px; word-break:break-all; }}
  .disclaimer    {{ background:#fef2f2; border-left:4px solid #f87171; padding:12px 16px;
                    border-radius:4px; font-size:13px; color:#7f1d1d; margin-top:16px; }}
  .footer        {{ background:#f8fafc; padding:16px 36px; font-size:12px; color:#94a3b8;
                    border-top:1px solid #e2e8f0; text-align:center; }}
</style>
</head>
<body>
<div class="wrapper">
  <div class="header">
    <div style="flex:1;">
      <h1>🩺 DermAI — Doctor Verification Required</h1>
      <p>A new patient case needs your review — {report_date}</p>
    </div>
    <div class="badge">PENDING REVIEW</div>
  </div>
  <div class="info-bar">
    <span><strong>Patient:</strong> {name}</span>
    <span><strong>Age:</strong> {age}</span>
    <span><strong>Email:</strong> {email}</span>
    {"<span><strong>Phone:</strong> " + phone + "</span>" if phone else ""}
  </div>
  <div class="body">
    <p style="font-size:14px;color:#374151;margin-top:0;">
      Dear Doctor,<br><br>
      A patient has submitted a skin condition image to <strong>DermAI</strong>.
      Please review the AI analysis and click <strong>Approve &amp; Send Prescription</strong>.
    </p>
    <div class="sec">📋 AI Analysis Findings</div>
    <table>
      <tr><td class="lbl">Predicted Condition</td><td class="val">{disease}</td></tr>
      <tr><td class="lbl">AI Confidence</td><td class="val">{confidence}%</td></tr>
      <tr><td class="lbl">Symptom Alignment</td><td class="val">{match_score}</td></tr>
      <tr><td class="lbl">Symptoms Reported</td><td class="val">{symptoms_text or "None"}</td></tr>
      <tr><td class="lbl">Matching Symptoms</td><td class="val">{matching_str}</td></tr>
      <tr><td class="lbl">Additional Typical Symptoms</td>
          <td class="val" style="color:#6b7280;font-weight:400;">{missing_str}</td></tr>
    </table>
    <div class="sec">💊 AI-Suggested Treatment Protocol</div>
    {medicine_html}
    <div class="action-box">
      <h3>✅ Approve &amp; Send Final Prescription to Patient</h3>
      <p>Click below to confirm the treatment and email the verified prescription to the patient.</p>
      <a href="{verify_url}" class="approve-btn">✅ &nbsp;Approve &amp; Send to Patient</a>
      <div class="url-fallback">Or copy this link: {verify_url}</div>
    </div>
    <div class="disclaimer">
      🔒 This link is unique to this patient case. Do not forward it.
    </div>
  </div>
  <div class="footer">
    <strong>DermAI</strong> — Internal Doctor Portal &nbsp;|&nbsp; Confidential
  </div>
</div>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
#  EMAIL 3 — Final verified prescription to patient (sent after doctor approves)
# ─────────────────────────────────────────────────────────────────────────────
def build_verified_prescription_html(name, age, email, phone, symptoms_text,
                                      disease, confidence, meds, doctor_notes):
    report_date   = datetime.now().strftime("%B %d, %Y  %H:%M")
    medicine_html = build_medicine_html_block(meds)

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<style>
  body      {{ font-family:'Segoe UI',Arial,sans-serif; background:#f4f4f4; margin:0; padding:0; }}
  .wrapper  {{ max-width:700px; margin:30px auto; background:#fff; border-radius:10px;
               box-shadow:0 4px 18px rgba(0,0,0,.10); overflow:hidden; }}
  .header   {{ background:linear-gradient(135deg,#0d9488,#0a7a70); padding:28px 36px; color:#fff; }}
  .header h1 {{ font-size:22px; margin:0 0 4px; }}
  .header p  {{ margin:0; font-size:13px; opacity:.85; }}
  .verified-strip {{ background:#f0fdf4; border-bottom:2px solid #86efac;
                     padding:14px 36px; display:flex; align-items:center; gap:12px; }}
  .verified-strip span {{ font-size:14px; font-weight:700; color:#166534; }}
  .info-bar  {{ background:#f0fdfa; padding:14px 36px; border-bottom:1px solid #ccf0ec; }}
  .info-bar span {{ font-size:13px; color:#374151; margin-right:28px; }}
  .info-bar strong {{ color:#0d9488; }}
  .body      {{ padding:28px 36px; }}
  .sec       {{ font-size:13px; font-weight:700; text-transform:uppercase;
                letter-spacing:1.2px; color:#0d9488; margin:24px 0 12px;
                border-bottom:1px solid #e5f4f3; padding-bottom:4px; }}
  .doc-notes {{ background:#eff6ff; border-left:4px solid #3b82f6; border-radius:6px;
                padding:14px 18px; font-size:14px; color:#1e3a8a; margin:16px 0; }}
  .disclaimer {{ background:#fff7ed; border-left:4px solid #f59e0b; padding:12px 16px;
                 border-radius:4px; font-size:13px; color:#92400e; margin-top:20px; }}
  .footer    {{ background:#f0fdfa; padding:18px 36px; font-size:12px; color:#6b7280;
                border-top:1px solid #ccf0ec; text-align:center; }}
  .footer strong {{ color:#0d9488; }}
</style>
</head>
<body>
<div class="wrapper">
  <div class="header">
    <h1>🩺 DermAI — Verified Prescription</h1>
    <p>Doctor-verified report issued on {report_date}</p>
  </div>
  <div class="verified-strip">
    <span style="font-size:22px;">✅</span>
    <span>This prescription has been reviewed and approved by a qualified dermatologist.</span>
  </div>
  <div class="info-bar">
    <span><strong>Patient:</strong> {name}</span>
    <span><strong>Age:</strong> {age}</span>
    <span><strong>Condition:</strong> {disease}</span>
  </div>
  <div class="body">
    <p style="font-size:15px;color:#374151;margin-top:0;">
      Dear <strong>{name}</strong>,<br><br>
      A qualified dermatologist has reviewed your DermAI analysis.
      Your <strong>final verified prescription</strong> is below.
    </p>
    {"<div class='sec'>👨‍⚕️ Doctor's Notes</div><div class='doc-notes'>" + doctor_notes + "</div>" if doctor_notes else ""}
    <div class="sec">📋 Confirmed Diagnosis</div>
    <p style="font-size:16px;font-weight:700;color:#0d9488;margin:0 0 4px;">{disease}</p>
    <p style="font-size:14px;color:#6b7280;margin:0;">AI Confidence: {confidence}%</p>
    <div class="sec">💊 Approved Treatment Protocol</div>
    {medicine_html}
    <div class="sec">🛡️ General Skin Care Tips</div>
    <ul style="padding-left:20px;color:#374151;font-size:14px;line-height:2.0;">
      <li>Keep the affected area clean and dry at all times.</li>
      <li>Avoid scratching, rubbing, or picking at the skin.</li>
      <li>Use gentle, fragrance-free moisturizers if dryness is present.</li>
      <li>Apply broad-spectrum SPF 30+ sunscreen every morning.</li>
      <li>Schedule a follow-up if symptoms worsen.</li>
    </ul>
    <div class="disclaimer">
      ⚠️ Follow the exact dosage and duration prescribed. Do not alter the treatment
      without consulting your doctor. Seek medical attention immediately for adverse effects.
    </div>
  </div>
  <div class="footer">
    <strong>DermAI</strong> — Doctor-Verified Prescription &nbsp;|&nbsp; Issued on {report_date}<br>
    Keep this email for your medical records.
  </div>
</div>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')


@app.route('/predict', methods=['POST'])
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

        # Predict
        img_bytes     = file.read()
        processed_img = preprocess_image(img_bytes)
        predictions   = model.predict(processed_img)[0]
        top_idx       = int(np.argmax(predictions))
        confidence    = float(predictions[top_idx] * 100)
        disease       = class_names[top_idx]

        # Symptom matching
        known = DISEASE_SYMPTOMS.get(disease, [])
        matching = [s for s in user_symptoms if any(s.lower() == k.lower() for k in known)]
        missing  = [k for k in known if not any(k.lower() == u.lower() for u in user_symptoms)]
        match_score = (
            f"{len(matching)} of {len(known)} typical symptoms match"
            if known else "No symptom data available"
        )
        confidence_str = f"{confidence:.2f}"
        meds = get_medicine_info(disease)

        # Build doctor verification token & URL
        token      = make_token(user_email, disease)
        verify_url = f"{BASE_URL}/doctor/verify?token={token}"

        # Store case for doctor portal
        case_data = {
            "name":          user_name,
            "age":           user_age,
            "email":         user_email,
            "phone":         user_phone,
            "symptoms_text": symptoms_text,
            "disease":       disease,
            "confidence":    confidence_str,
            "match_score":   match_score,
            "matching":      matching,
            "missing":       missing,
            "meds":          meds,
            "verified":      False,
        }
        store_case(token, case_data)

        # Send patient email
        patient_sent = False
        if user_email:
            patient_sent = send_via_smtp(
                to      = user_email,
                subject = f"DermAI Report — {disease} (Awaiting Doctor Verification)",
                html    = build_patient_email_html(
                              user_name, user_age, user_email, user_phone, symptoms_text,
                              disease, confidence_str, match_score, matching, missing, meds),
                text    = build_patient_email_plain(
                              user_name, user_age, symptoms_text,
                              disease, confidence_str, match_score, matching, missing, meds),
            )

        # Send doctor verification email
        doctor_sent = send_via_smtp(
            to      = DOCTOR_EMAIL,
            subject = f"[DermAI] New Case — {user_name} | {disease} | Verification Required",
            html    = build_doctor_email_html(
                          user_name, user_age, user_email, user_phone, symptoms_text,
                          disease, confidence_str, match_score, matching, missing,
                          meds, verify_url),
            text    = (
                f"New DermAI patient case requires your review.\n\n"
                f"Patient: {user_name}, Age: {user_age}\n"
                f"Predicted: {disease} ({confidence_str}% confidence)\n\n"
                f"Verify and send prescription:\n{verify_url}"
            ),
        )

        return jsonify({
            "disease":         disease,
            "confidence":      confidence_str,
            "match_score":     match_score,
            "matching":        matching,
            "missing":         missing,
            "medicines":       meds,
            "email_sent":      patient_sent,
            "doctor_notified": doctor_sent,
        })

    except Exception as exc:
        print("Predict error:", exc)
        return jsonify({"error": "Prediction failed"}), 500


# ─────────────────────────────────────────────────────────────────────────────
#  Doctor Portal  — /doctor/verify?token=...
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/doctor/verify', methods=['GET', 'POST'])
def doctor_verify():
    token = request.args.get("token", "")
    case  = get_case(token)

    if not case:
        return (
            "<h2 style='font-family:sans-serif;color:#dc2626;padding:40px;'>"
            "❌ Invalid or expired verification link.</h2>", 404
        )

    if case.get("verified"):
        return (
            "<h2 style='font-family:sans-serif;color:#0d9488;padding:40px;'>"
            "✅ This case has already been verified and the prescription sent.</h2>", 200
        )

    if request.method == 'POST':
        doctor_notes = request.form.get("doctor_notes", "").strip()

        send_via_smtp(
            to      = case["email"],
            subject = f"DermAI — ✅ Your Verified Prescription is Ready ({case['disease']})",
            html    = build_verified_prescription_html(
                          case["name"], case["age"], case["email"], case["phone"],
                          case["symptoms_text"], case["disease"], case["confidence"],
                          case["meds"], doctor_notes),
            text    = (
                f"Dear {case['name']},\n\n"
                f"Your DermAI case for {case['disease']} has been reviewed and approved "
                f"by a qualified dermatologist.\n\n"
                f"Doctor Notes: {doctor_notes or 'None'}\n\n"
                "Please check the full HTML email for your treatment protocol.\n\n"
                "DermAI — Doctor-Verified Prescription"
            ),
        )

        case["verified"] = True
        PENDING_CASES[token] = case

        return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
  body {{ font-family:'Segoe UI',sans-serif; background:#f0fdf4; display:flex;
         align-items:center; justify-content:center; min-height:100vh; margin:0; }}
  .box {{ background:#fff; border-radius:12px; padding:48px 56px; text-align:center;
          box-shadow:0 8px 32px rgba(0,0,0,.10); max-width:480px; }}
  h2   {{ color:#166534; margin:0 0 12px; }}
  p    {{ color:#374151; font-size:15px; }}
</style></head>
<body>
<div class="box">
  <div style="font-size:56px;margin-bottom:16px;">✅</div>
  <h2>Prescription Sent!</h2>
  <p>The verified prescription has been emailed to<br>
     <strong>{case['email']}</strong>.</p>
  <p style="margin-top:20px;color:#6b7280;font-size:13px;">
    Patient: {case['name']} &nbsp;|&nbsp; Condition: {case['disease']}</p>
</div>
</body></html>"""

    # GET — show Doctor Portal form
    med_html     = build_medicine_html_block(case["meds"])
    matching_str = ", ".join(case["matching"]) if case["matching"] else "None"
    missing_str  = ", ".join(case["missing"])  if case["missing"]  else "None"

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DermAI — Doctor Portal</title>
<style>
  :root {{ --teal:#0d9488; --teal-dark:#0a7a70; --ink:#0f172a;
           --muted:#64748b; --border:#e2e8f0; --bg:#f8fafc; }}
  *, *::before, *::after {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Segoe UI',Arial,sans-serif; background:var(--bg); color:var(--ink); }}
  .navbar {{ background:#1e293b; padding:0 40px; height:60px; display:flex;
             align-items:center; gap:12px; }}
  .navbar span {{ color:#fff; font-size:18px; font-weight:700; }}
  .navbar em   {{ color:#94a3b8; font-size:13px; font-style:normal; }}
  .hero {{ background:linear-gradient(135deg,#0d9488,#0a7a70); color:#fff;
           padding:40px 40px 32px; }}
  .hero h1 {{ font-size:26px; margin-bottom:6px; }}
  .hero p  {{ font-size:14px; opacity:.85; }}
  .container {{ max-width:820px; margin:32px auto; padding:0 24px 64px; }}
  .card {{ background:#fff; border:1px solid var(--border); border-radius:12px;
           box-shadow:0 2px 12px rgba(15,23,42,.07); margin-bottom:24px; overflow:hidden; }}
  .card-head {{ padding:18px 26px; border-bottom:1px solid var(--border);
                background:#fafafa; font-weight:700; font-size:15px;
                display:flex; align-items:center; gap:10px; }}
  .card-body {{ padding:22px 26px; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; }}
  td    {{ padding:9px 8px; vertical-align:top; }}
  td.lbl {{ width:40%; color:var(--muted); }}
  td.val {{ font-weight:600; color:var(--ink); }}
  tr:nth-child(even) td {{ background:#f9fafb; }}
  textarea {{ width:100%; min-height:120px; border:1.5px solid var(--border);
              border-radius:8px; padding:12px 14px; font-size:14px;
              font-family:inherit; resize:vertical; outline:none; }}
  textarea:focus {{ border-color:var(--teal); box-shadow:0 0 0 3px rgba(13,148,136,.1); }}
  .approve-btn {{ display:block; width:100%; background:var(--teal); color:#fff;
                  border:none; border-radius:10px; padding:16px;
                  font-size:16px; font-weight:700; cursor:pointer;
                  transition:all 0.2s; letter-spacing:0.4px; }}
  .approve-btn:hover {{ background:var(--teal-dark); transform:translateY(-1px); }}
  .warning-box {{ background:#fef2f2; border-left:4px solid #f87171; border-radius:6px;
                  padding:12px 16px; font-size:13px; color:#7f1d1d; margin-top:16px; }}
</style>
</head>
<body>
<nav class="navbar">
  <span>🩺 DermAI</span><em>— Doctor Verification Portal</em>
</nav>
<div class="hero">
  <h1>Patient Case Review</h1>
  <p>Review the AI analysis and approve to send the verified prescription to the patient.</p>
</div>
<div class="container">
  <div class="card">
    <div class="card-head">👤 Patient Information</div>
    <div class="card-body">
      <table>
        <tr><td class="lbl">Full Name</td><td class="val">{case['name']}</td></tr>
        <tr><td class="lbl">Age</td><td class="val">{case['age']}</td></tr>
        <tr><td class="lbl">Email</td><td class="val">{case['email']}</td></tr>
        <tr><td class="lbl">Phone</td><td class="val">{case['phone'] or "N/A"}</td></tr>
        <tr><td class="lbl">Symptoms Reported</td>
            <td class="val">{case['symptoms_text'] or "None"}</td></tr>
      </table>
    </div>
  </div>
  <div class="card">
    <div class="card-head">🔬 AI Analysis Findings</div>
    <div class="card-body">
      <table>
        <tr><td class="lbl">Predicted Condition</td>
            <td class="val" style="color:var(--teal);font-size:16px;">{case['disease']}</td></tr>
        <tr><td class="lbl">AI Confidence</td><td class="val">{case['confidence']}%</td></tr>
        <tr><td class="lbl">Symptom Alignment</td><td class="val">{case['match_score']}</td></tr>
        <tr><td class="lbl">Matching Symptoms</td><td class="val">{matching_str}</td></tr>
        <tr><td class="lbl">Additional Typical Symptoms</td>
            <td class="val" style="color:var(--muted);font-weight:400;">{missing_str}</td></tr>
      </table>
    </div>
  </div>
  <div class="card">
    <div class="card-head">💊 AI-Suggested Treatment Protocol</div>
    <div class="card-body">{med_html}</div>
  </div>
  <div class="card">
    <div class="card-head">✅ Doctor Approval</div>
    <div class="card-body">
      <form method="POST" action="/doctor/verify?token={token}">
        <div style="margin-bottom:16px;">
          <label style="display:block;font-size:14px;font-weight:600;
                        color:var(--ink);margin-bottom:8px;">
            Doctor's Notes / Modifications
            <span style="color:var(--muted);font-weight:400;">(optional)</span>
          </label>
          <textarea name="doctor_notes"
            placeholder="Add any notes, dosage changes, or additional advice..."></textarea>
        </div>
        <button type="submit" class="approve-btn">
          ✅ &nbsp;Approve &amp; Send Verified Prescription to Patient
        </button>
      </form>
      <div class="warning-box" style="margin-top:16px;">
        🔒 This will immediately email the verified prescription to
        <strong>{case['email']}</strong>. This cannot be undone.
      </div>
    </div>
  </div>
</div>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
#  Run
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"\nDermAI server starting on http://127.0.0.1:{PORT}\n")
    app.run(host='0.0.0.0', port=PORT, debug=False)