import streamlit as st
from model import analyze_text
from explain import explain, risk_level
from news import get_news
from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="Trust & Safety AI v5.1", layout="centered")

st.title("🧠 AI Trust & Safety System v5.1")

# ---------------- ANALYSIS ----------------
text = st.text_area("Enter text")

if st.button("Analyze"):
    if text.strip():

        result = analyze_text(text)

        st.write("## 🧠 Risk Overview")

        st.write("Bias Risk:", risk_level(result["bias_score"]))
        st.write("Toxicity Risk:", risk_level(result["toxicity"]))

        st.write("### 📊 Raw Metrics")
        st.json(result)

        st.write("### 🔍 Explanation")
        st.info(explain(result, text))

# ---------------- NEWS FILTERING ----------------
st.subheader("🌍 Relevant News")

news = get_news()

if text.strip():
    input_vec = embedder.encode(text, convert_to_tensor=True)

    filtered = []

    for n in news:
        vec = embedder.encode(n["title"], convert_to_tensor=True)
        score = util.cos_sim(input_vec, vec).item()

        if score > 0.25:
            n["relevance"] = round(score, 3)
            filtered.append(n)

    filtered = sorted(filtered, key=lambda x: x["relevance"], reverse=True)

    for n in filtered:
        st.write("## 🧾", n["title"])
        st.write(n["summary"])
        st.write("Relevance:", n["relevance"])
        st.write("[Read]", n["link"])
        st.divider()
