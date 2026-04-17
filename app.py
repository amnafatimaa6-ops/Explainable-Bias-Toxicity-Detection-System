import streamlit as st
from model import analyze_text
from news import get_news

st.set_page_config(page_title="AI Ethics Radar Semantic", layout="centered")

st.title("🧠 AI Ethics Radar — Semantic AI System")

# ---------------- INPUT ----------------
st.subheader("🔍 Analyze Text")

text = st.text_area("Enter text")

if st.button("Analyze"):
    if text.strip():
        result = analyze_text(text)

        st.write("### 🧠 AI Analysis")

        st.write("**Category:**", result["category"])
        st.write("**Bias Score (Semantic):**", result["bias_score"])
        st.write("**Toxicity Score:**", result["toxicity"])
        st.write("**Violence Score:**", result["violence_score"])
        st.write("**News Score:**", result["news_score"])
        st.write("**Sentiment:**", result["sentiment"])
        st.write("**Explanation:**", result["explanation"])

# ---------------- NEWS ----------------
st.subheader("🌍 Live News Scan")

articles = get_news()

for a in articles:
    st.write("## 🧾 News")
    st.write(a["title"])
    st.write(a["summary"])

    result = analyze_text(a["title"])

    st.write("### 🔍 AI Analysis")
    st.write("**Category:**", result["category"])
    st.write("**Bias Score:**", result["bias_score"])
    st.write("**Toxicity:**", result["toxicity"])
    st.write("**Explanation:**", result["explanation"])

    st.write("[Read full article]", a["link"])
    st.divider()
