import streamlit as st
from model import analyze_text
from explain import explain, risk_level
from news import get_news
from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="AI Ethics Radar v8", layout="centered")

st.title("🧠 AI Ethics Radar v8 — Research-Grade Trust System")

text = st.text_area("Enter text")

# ---------------- USER ANALYSIS ----------------
if st.button("Analyze") and text.strip():

    result = analyze_text(text)

    st.subheader("🧠 Risk Overview")
    st.write("Bias:", risk_level(result["bias_score"]))
    st.write("Toxicity:", risk_level(result["toxicity"]))
    st.write("Violence:", risk_level(result["violence_score"]))

    st.subheader("📊 Raw Metrics")
    st.json(result)

    st.subheader("🔍 Explanation")
    st.info(explain(result, text))

# ---------------- NEWS ----------------
st.subheader("🌍 News Intelligence Layer v8")

news = get_news()

if text.strip():

    input_vec = embedder.encode(text, convert_to_tensor=True)

    scored_news = []

    for n in news:

        full_text = n["title"] + " " + n["summary"]

        news_vec = embedder.encode(full_text, convert_to_tensor=True)
        relevance = util.cos_sim(input_vec, news_vec).item()

        news_result = analyze_text(full_text)

        risk_score = (
            news_result["toxicity"] * 0.25 +
            news_result["bias_score"] * 0.25 +
            news_result["violence_score"] * 0.2 +
            relevance * 0.3
        )

        if risk_score > 0.75:
            label = "🔴 High Risk News"
        elif risk_score > 0.5:
            label = "🟠 Sensitive News"
        elif risk_score > 0.25:
            label = "🟡 Contextual News"
        else:
            label = "🟢 Informational"

        n.update({
            "relevance": round(relevance, 3),
            "risk_score": round(risk_score, 3),
            "risk_label": label
        })

        scored_news.append(n)

    scored_news = sorted(scored_news, key=lambda x: x["risk_score"], reverse=True)

    for n in scored_news:

        st.write("## 🧾", n["title"])
        st.write(n["summary"])

        st.write("📌 Relevance:", n["relevance"])
        st.write("⚠️ Risk Score:", n["risk_score"])
        st.write("🚨 Status:", n["risk_label"])

        st.write("[Read]", n["link"])
        st.divider()

else:
    st.info("Enter text to activate intelligence system")
