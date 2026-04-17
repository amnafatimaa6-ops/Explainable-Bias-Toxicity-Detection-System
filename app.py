import streamlit as st
from model import analyze_text
from explain import explain, risk_level
from news import get_news
from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="AI Trust Safety v5.2", layout="centered")

st.title("🧠 AI Ethics Radar v5.2")

# ---------------- INPUT ----------------
text = st.text_area("Enter text")

# ---------------- ANALYSIS ----------------
if st.button("Analyze") and text.strip():

    result = analyze_text(text)

    st.subheader("🧠 Risk Overview")
    st.write("Bias Risk:", risk_level(result["bias_score"]))
    st.write("Toxicity Risk:", risk_level(result["toxicity"]))

    st.subheader("📊 Raw Metrics")
    st.json(result)

    st.subheader("🔍 Explanation")
    st.info(explain(result, text))

# ---------------- NEWS ----------------
st.subheader("🌍 News Intelligence Layer")

news = get_news()

if text.strip():

    input_vec = embedder.encode(text, convert_to_tensor=True)

    scored_news = []

    for n in news:
        full_text = n["title"] + " " + n["summary"]

        vec = embedder.encode(full_text, convert_to_tensor=True)
        score = util.cos_sim(input_vec, vec).item()

        # 🔥 FIX: NEVER EMPTY SCREEN
        if score > 0.12:
            n["relevance"] = round(score, 3)
            scored_news.append(n)

    # fallback guarantee
    if len(scored_news) == 0:
        scored_news = news[:3]
        for n in scored_news:
            n["relevance"] = 0.0

    scored_news = sorted(scored_news, key=lambda x: x["relevance"], reverse=True)

    for n in scored_news:
        st.write("## 🧾", n["title"])
        st.write(n["summary"])
        st.write("Relevance:", n["relevance"])
        st.write("[Read]", n["link"])
        st.divider()

else:
    st.info("Type text to activate news intelligence layer")
