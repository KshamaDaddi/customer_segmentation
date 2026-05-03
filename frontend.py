import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Customer Intelligence", layout="wide")
st.title("🚀 AI Customer Intelligence Platform")

BASE_URL = "http://127.0.0.1:8000"

# ── Helper: display AI response ───────────────────────────────────────────────
def show_insights(response_json: dict):
    insights = response_json.get("insights")
    error    = response_json.get("error")
    fallback = response_json.get("fallback")

    if insights:
        st.info(insights)
    elif fallback:
        st.warning(f"⚠️ Gemini unavailable ({error}). Showing fallback recommendations:")
        for k, v in fallback.items():
            st.write(f"**{k.title()}:** {v}")
    else:
        st.error(f"❌ AI error: {error}")

# ── Helper: check if backend is alive ─────────────────────────────────────────
@st.cache_data(ttl=5)
def backend_alive():
    try:
        r = requests.get(f"{BASE_URL}/docs", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

if not backend_alive():
    st.error(
        "❌ **Backend is not running.**  \n"
        "Open a terminal in your project folder and run:  \n"
        "```\npython backend.py\n```"
    )
    st.stop()   # halt rendering — no point showing broken UI

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – Predict Customer Segment
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("🔎 Analyze a Customer")

col1, col2, col3 = st.columns(3)
with col1:
    age      = st.slider("Age", 18, 60, 25)
with col2:
    income   = st.slider("Annual Income (₹)", 20_000, 120_000, 50_000, step=1_000)
with col3:
    spending = st.slider("Spending Score (1–100)", 1, 100, 50)

if st.button("🔍 Analyze Customer"):
    with st.spinner("Predicting segment…"):
        try:
            res = requests.post(
                f"{BASE_URL}/predict",
                params={"age": age, "income": income, "spending_score": spending},
                timeout=10,
            )
            if res.status_code == 200:
                data    = res.json()
                segment = data.get("segment_name", "Unknown")
                cluster = data.get("cluster_id", "?")

                st.success(f"🎯 Segment: **{segment}**  |  Cluster ID: {cluster}")

                st.subheader("🧠 AI Business Insights")
                with st.spinner("Getting Gemini analysis…"):
                    ai = requests.post(
                        f"{BASE_URL}/ai-insights",
                        params={"segment_name": segment},
                        timeout=30,
                    )
                    if ai.status_code == 200:
                        show_insights(ai.json())
                    else:
                        st.warning(f"⚠️ AI service returned HTTP {ai.status_code}")
            else:
                st.error(f"❌ Prediction failed (HTTP {res.status_code}): {res.text}")

        except Exception as exc:
            st.error(f"❌ Error: {exc}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – Live Dashboard  (reads from data/live_data.csv via /data API)
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("📊 Live Customer Dashboard")

@st.cache_data(ttl=10)
def load_live_data():
    try:
        r = requests.get(f"{BASE_URL}/data", timeout=5)
        if r.status_code == 200:
            records = r.json().get("data", [])
            return pd.DataFrame(records) if records else pd.DataFrame()
    except Exception:
        pass
    return pd.DataFrame()

df = load_live_data()

if df.empty:
    st.info("No data yet — click **Analyze Customer** above to generate predictions.")
else:
    # KPI row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Customers", len(df))
    m2.metric("Premium",    int((df["segment"] == "Premium").sum()))
    m3.metric("High Value", int((df["segment"] == "High Value").sum()))
    m4.metric("Low Value",  int((df["segment"] == "Low Value").sum()))

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.scatter(
            df, x="income", y="spending_score", color="segment",
            title="Income vs Spending Score by Segment",
            labels={"income": "Income (₹)", "spending_score": "Spending Score"},
        )
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = px.histogram(
            df, x="segment", color="segment",
            title="Segment Count",
            category_orders={
                "segment": ["Low Value", "Mid Value", "High Value", "Premium"]
            },
        )
        st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – AI Strategy Chat
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("💬 AI Strategy Assistant")

query = st.text_area(
    "Ask a business question",
    placeholder="e.g. How do I retain Mid Value customers who are at churn risk?",
)

if st.button("Ask AI"):
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Thinking…"):
            try:
                ai = requests.post(
                    f"{BASE_URL}/ai-chat",          # ← dedicated endpoint
                    params={"question": query},
                    timeout=30,
                )
                if ai.status_code == 200:
                    show_insights(ai.json())
                else:
                    st.error(f"❌ Backend returned HTTP {ai.status_code}: {ai.text}")
            except Exception as exc:
                st.error(f"❌ Error: {exc}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – Download Data
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("⬇️ Download Data")

if not df.empty:
    st.download_button(
        label="📥 Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="customer_segments.csv",
        mime="text/csv",
    )
else:
    st.info("No data to download yet.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
with st.expander("📊 How to connect Power BI"):
    st.markdown("""
**Step-by-step:**
1. Open **Power BI Desktop**
2. Click **Home → Get Data → Web**
3. Paste this URL:
```
http://127.0.0.1:8000/powerbi
```
4. Click **OK → Load**
5. Your table will appear with columns: `age`, `income`, `spending_score`, `segment`, `timestamp`
6. To refresh data: click **Home → Refresh**

> ⚠️ Make sure `backend.py` is running whenever you refresh in Power BI.
""")