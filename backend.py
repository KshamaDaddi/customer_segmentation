from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import joblib
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime

# ── Load .env ──────────────────────────────────────────────────────────────────
load_dotenv()

app = FastAPI(title="AI Customer Intelligence API")

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")
DATA_PATH  = os.path.join(DATA_DIR, "live_data.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# ── Load ML models ─────────────────────────────────────────────────────────────
for path, name in [
    (os.path.join(MODELS_DIR, "scaler.pkl"), "scaler.pkl"),
    (os.path.join(MODELS_DIR, "kmeans.pkl"), "kmeans.pkl"),
]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found at: {path}")

scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
kmeans = joblib.load(os.path.join(MODELS_DIR, "kmeans.pkl"))
print(f"✅  Models loaded  →  {MODELS_DIR}")
print(f"✅  Data path      →  {DATA_PATH}")

# ── Segment mapping ────────────────────────────────────────────────────────────
SEGMENTS = {0: "Low Value", 1: "Mid Value", 2: "High Value", 3: "Premium"}

# ══════════════════════════════════════════════════════════════════════════════
#  FREE AI  —  Groq  (100% free, no credit card, 14,400 req/day)
#  Sign up at console.groq.com  →  create API key  →  paste in .env
# ══════════════════════════════════════════════════════════════════════════════

# Models available on Groq free tier (in priority order)
GROQ_MODELS = [
    "llama-3.3-70b-versatile",     # best quality
    "llama-3.1-8b-instant",        # fastest
    "mixtral-8x7b-32768",          # good for long prompts
    "gemma2-9b-it",                # Google Gemma, also free
]

def _fallback_response():
    return {
        "strategy":   "Run targeted campaigns for this segment",
        "retention":  "Offer loyalty discounts",
        "engagement": "Personalise email & push notifications",
    }

def _ok(text: str, model: str = ""):
    return {"insights": text, "error": None, "fallback": None, "model_used": model}

def _err(msg: str):
    print(f"⚠️  AI error: {msg}")
    return {"insights": None, "error": msg, "fallback": _fallback_response()}


def _call_ai(prompt: str):
    """
    Uses Groq free API — always returns a dict, never raises.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key:
        return _err(
            "GROQ_API_KEY not found in .env file. "
            "Get a FREE key at console.groq.com (no credit card needed)."
        )

    print(f"Groq key loaded: {api_key[:8]}…")

    try:
        from groq import Groq
    except ImportError:
        return _err(
            "groq package not installed. Run: pip install groq"
        )

    client     = Groq(api_key=api_key)
    last_error = {}

    for model_name in GROQ_MODELS:
        try:
            completion = client.chat.completions.create(
                model    = model_name,
                messages = [
                    {
                        "role":    "system",
                        "content": "You are a senior business analyst. Be concise and structured.",
                    },
                    {
                        "role":    "user",
                        "content": prompt,
                    },
                ],
                temperature = 0.7,
                max_tokens  = 1024,
            )
            text = completion.choices[0].message.content
            print(f"✅  Groq response received — model: {model_name}")
            return _ok(text, model=model_name)

        except Exception as e:
            last_error[model_name] = str(e)
            print(f"  {model_name} failed: {e}")
            continue

    # All models failed
    summary = " | ".join(f"{m}: {e}" for m, e in last_error.items())
    return _err(f"All Groq models failed → {summary}")


# ── /predict ───────────────────────────────────────────────────────────────────
@app.post("/predict")
def predict(age: int, income: int, spending_score: int):
    try:
        X        = np.array([[age, income, spending_score]])
        X_scaled = scaler.transform(X)
        cluster  = int(kmeans.predict(X_scaled)[0])
        segment  = SEGMENTS.get(cluster, "Unknown")

        row = pd.DataFrame([{
            "age":            age,
            "income":         income,
            "spending_score": spending_score,
            "segment":        segment,
            "timestamp":      datetime.now().isoformat(),
        }])

        if os.path.exists(DATA_PATH):
            row.to_csv(DATA_PATH, mode="a", header=False, index=False)
        else:
            row.to_csv(DATA_PATH, index=False)

        return {"segment_name": segment, "cluster_id": cluster}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── /ai-insights ───────────────────────────────────────────────────────────────
@app.post("/ai-insights")
def ai_insights(segment_name: str):
    prompt = f"""Customer Segment: {segment_name}

Provide a structured business analysis with these 5 sections:

1. Customer Behavior — Who are they? What do they do?
2. Marketing Strategy — How to target and reach them?
3. Product Recommendations — What to offer them?
4. Revenue Impact — What is the business value of this segment?
5. Risk Level — What risks exist with this segment?

Keep each section to 2-3 sentences. Be practical and specific.
"""
    return _call_ai(prompt)


# ── /ai-chat ───────────────────────────────────────────────────────────────────
@app.post("/ai-chat")
def ai_chat(question: str):
    prompt = f"""Business Question: {question}

Give a clear, concise, actionable answer in plain English.
If relevant, include specific steps or examples.
"""
    return _call_ai(prompt)


# ── /data ──────────────────────────────────────────────────────────────────────
@app.get("/data")
def get_data():
    if not os.path.exists(DATA_PATH):
        return {"data": [], "message": "No predictions yet."}
    df = pd.read_csv(DATA_PATH)
    return {"data": df.to_dict(orient="records")}


# ── /powerbi  ──────────────────────────────────────────────────────────────────
# Power BI → Get Data → Web → http://127.0.0.1:8000/powerbi
@app.get("/powerbi")
def powerbi_csv():
    if not os.path.exists(DATA_PATH):
        csv_content = "age,income,spending_score,segment,timestamp\n"
    else:
        df = pd.read_csv(DATA_PATH)
        csv_content = df.to_csv(index=False)

    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "inline; filename=live_data.csv"},
    )


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)