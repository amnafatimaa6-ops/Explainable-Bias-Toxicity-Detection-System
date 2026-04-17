import streamlit as st
from model import analyze_text
from explain import explain_result
from news import get_news

st.set_page_config(page_title="AI Ethics Radar v3", layout="centered")

st.title("🧠 AI Ethics Radar v3 — Explainable AI System")

# ---------------- INPUT ----------------
text = st.text_area("Enter text")

if st.button("Analyze"):
    if text.strip():

        result = analyze_text(text)
        explanation = explain_result(result)

        st.write("## 🧠 AI Analysis")

        st.write("**Category:**", result["category"])
        st.write("**Bias Type:**", result["bias_type"])
        st.write("**Bias Score:**", result["bias_score"])
        st.write("**Toxicity:**", result["toxicity"])
        st.write("**Violence Score:**", result["violence_score"])
        st.write("**News Score:**", result["news_score"])
        st.write("**Sentiment:**", result["sentiment"])

        st.write("### 🧾 Explanation Chain")
        st.info(explanation)

# ---------------- NEWS ----------------
st.subheader("🌍 Live News Scan")

articles = get_news()

for a in articles:
    st.write("## 🧾 News")
    st.write(a["title"])
    st.write(a["summary"])

    result = analyze_text(a["title"])
    explanation = explain_result(result)

    st.write("### 🔍 AI Analysis")
    st.write(result["category"])
    st.write("Bias:", result["bias_score"])
    st.write("Toxicity:", result["toxicity"])

    st.write("**Why:**", explanation)

    st.write("[Read full article]", a["link"])
    st.divider()
