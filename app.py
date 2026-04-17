import streamlit as st
from model import analyze_text
from explain import explain, risk_level
from news import get_news

st.set_page_config(page_title="AI Ethics Radar v4", layout="centered")

st.title("🧠 AI Ethics Radar v4 — Explainable AI System")

# ---------------- INPUT ----------------
text = st.text_area("Enter text")

if st.button("Analyze"):
    if text.strip():

        result = analyze_text(text)

        st.write("## 🧠 AI Analysis")

        st.write("**Category:**", result["category"])
        st.write("**Bias Type:**", result["bias_type"])

        st.write("**Bias Level:**", risk_level(result["bias_score"]))
        st.write("**Toxicity Level:**", risk_level(result["toxicity"]))
        st.write("**Violence Level:**", risk_level(result["violence_score"]))
        st.write("**News Level:**", risk_level(result["news_score"]))

        st.write("### 📊 Raw Scores")
        st.json({
            "bias": result["bias_score"],
            "toxicity": result["toxicity"],
            "violence": result["violence_score"],
            "news": result["news_score"]
        })

        st.write("### 🧾 Explanation")
        st.info(explain(result, text))

# ---------------- NEWS ----------------
st.subheader("🌍 Live News Scan")

articles = get_news()

for a in articles:
    st.write("## 🧾 News")
    st.write(a["title"])
    st.write(a["summary"])

    result = analyze_text(a["title"])

    st.write("**Category:**", result["category"])
    st.write("Bias:", result["bias_score"])
    st.write("Toxicity:", result["toxicity"])

    st.write("**Why:**", risk_level(result["bias_score"]), "| system based classification")

    st.write("[Read full article]", a["link"])
    st.divider()
