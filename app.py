import streamlit as st
from model import analyze_text
from explain import explain, risk_level
from news import get_news
from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="AI Ethics Radar v6", layout="centered")

st.title("🧠 AI Ethics Radar v6 — Intelligence System")

# ---------------- USER INPUT ----------------
text = st.text_area("Enter text")

# ---------------- TEXT ANALYSIS ----------------
if st.button("Analyze") and text.strip():

    result = analyze_text(text)

    st.subheader("🧠 Risk Overview")
    st.write("Bias Risk:", risk_level(result["bias_score"]))
    st.write("Toxicity Risk:", risk_level(result["toxicity"]))

    st.subheader("📊 Raw Metrics")
    st.json(result)

    st.subheader("🔍 Explanation")
    st.info(explain(result, text))

# ---------------- NEWS INTELLIGENCE ----------------
st.subheader("🌍 Live AI News Intelligence Layer")

news = get_news()

if text.strip():

    input_vec = embedder.encode(text, convert_to_tensor=True)

    scored_news = []

    for n in news:

        full_text = n["title"] + " " + n["summary"]

        # 🔹 relevance
        news_vec = embedder.encode(full_text, convert_to_tensor=True)
        relevance = util.cos_sim(input_vec, news_vec).item()

        # 🔹 AI risk analysis on news itself
        news_result = analyze_text(full_text)

        risk_score = (
            news_result["toxicity"] * 0.4 +
            news_result["bias_score"] * 0.4 +
            news_result.get("violence_score", 0) * 0.2
        )

        # 🔹 risk label
        if risk_score > 0.75:
            risk_label = "🔴 High Risk News"
        elif risk_score > 0.5:
            risk_label = "🟠 Medium Risk News"
        elif risk_score > 0.25:
            risk_label = "🟡 Low Risk News"
        else:
            risk_label = "🟢 Safe News"

        n.update({
            "relevance": round(relevance, 3),
            "risk_score": round(risk_score, 3),
            "risk_label": risk_label
        })

        scored_news.append(n)

    # ---------------- SORTING ----------------
    scored_news = sorted(
        scored_news,
        key=lambda x: (x["risk_score"] + x["relevance"]),
        reverse=True
    )

    # ---------------- DISPLAY ----------------
    for n in scored_news:

        st.write("## 🧾", n["title"])
        st.write(n["summary"])

        st.write("📌 Relevance:", n["relevance"])
        st.write("⚠️ Risk Score:", n["risk_score"])
        st.write("🚨 Status:", n["risk_label"])

        st.write("[Read]", n["link"])
        st.divider()

else:
    st.info("Enter text to activate AI intelligence system")
