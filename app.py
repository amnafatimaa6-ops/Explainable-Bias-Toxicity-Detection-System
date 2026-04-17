import streamlit as st
from model import analyze_text
from explain import explain, risk_level
from news import get_news
from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="AI Ethics Radar v7", layout="centered")

st.title("🧠 AI Ethics Radar v7 — Context-Aware Safety System")

# ---------------- INPUT ----------------
text = st.text_area("Enter text")

# ---------------- USER ANALYSIS ----------------
if st.button("Analyze") and text.strip():

    result = analyze_text(text)

    st.subheader("🧠 Risk Overview")

    st.write("Bias Risk:", risk_level(result["bias_score"]))
    st.write("Toxicity Risk:", risk_level(result["toxicity"]))
    st.write("Violence Risk:", risk_level(result["violence_score"]))

    st.subheader("📊 Raw Metrics")
    st.json(result)

    st.subheader("🔍 Explanation")
    st.info(explain(result, text))

# ---------------- NEWS INTELLIGENCE ----------------
st.subheader("🌍 Live AI News Intelligence Layer")

news = get_news()

def compute_news_risk(news_result, full_text):

    text = full_text.lower()

    tox = news_result["toxicity"]
    bias = news_result["bias_score"]

    violence_keywords = [
        "kill", "attack", "war", "bomb", "shoot", "death",
        "arrest", "court", "crime", "police", "indictment"
    ]

    political_keywords = [
        "president", "government", "minister", "election", "vote"
    ]

    crisis_keywords = [
        "earthquake", "flood", "fire", "explosion", "disaster"
    ]

    v = sum(w in text for w in violence_keywords)
    p = sum(w in text for w in political_keywords)
    c = sum(w in text for w in crisis_keywords)

    domain_risk = (v * 0.25) + (p * 0.1) + (c * 0.35)

    risk_score = (tox * 0.4) + (bias * 0.3) + (domain_risk * 0.3)

    if risk_score > 0.75:
        label = "🔴 Critical News Risk"
    elif risk_score > 0.5:
        label = "🟠 High Attention News"
    elif risk_score > 0.25:
        label = "🟡 Moderate Context Risk"
    else:
        label = "🟢 Informational / Safe"

    reasons = []

    if v:
        reasons.append("Violence / conflict context detected")
    if p:
        reasons.append("Political content detected")
    if c:
        reasons.append("Crisis / disaster context detected")
    if tox > 0.6:
        reasons.append("High toxicity language detected")

    if not reasons:
        reasons.append("Neutral informational content")

    return risk_score, label, " | ".join(reasons)


if text.strip():

    input_vec = embedder.encode(text, convert_to_tensor=True)

    scored_news = []

    for n in news:

        full_text = n["title"] + " " + n["summary"]

        news_vec = embedder.encode(full_text, convert_to_tensor=True)
        relevance = util.cos_sim(input_vec, news_vec).item()

        news_result = analyze_text(full_text)

        risk_score, risk_label, explanation = compute_news_risk(news_result, full_text)

        n.update({
            "relevance": round(relevance, 3),
            "risk_score": round(risk_score, 3),
            "risk_label": risk_label,
            "explanation": explanation
        })

        scored_news.append(n)

    scored_news = sorted(
        scored_news,
        key=lambda x: (x["risk_score"] + x["relevance"]),
        reverse=True
    )

    for n in scored_news:

        st.write("## 🧾", n["title"])
        st.write(n["summary"])

        st.write("📌 Relevance:", n["relevance"])
        st.write("⚠️ Risk Score:", n["risk_score"])
        st.write("🚨 Status:", n["risk_label"])
        st.write("🧠 Explanation:", n["explanation"])

        st.write("[Read Article]", n["link"])
        st.divider()

else:
    st.info("Enter text to activate AI intelligence system")
