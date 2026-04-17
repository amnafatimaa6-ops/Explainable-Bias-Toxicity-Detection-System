import streamlit as st
from model import analyze_text, highlight_text
from news import get_news

st.set_page_config(page_title="AI Ethics Radar", layout="centered")

st.title("🧠 AI Ethics Radar — Stable Version")

# ---------------- TEXT INPUT ----------------
st.subheader("🔍 Analyze Text")

text = st.text_area("Enter text")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter text")
    else:
        result = analyze_text(text)

        st.write("### 🧠 AI Analysis")
        st.write("**Category:**", result["category"])
        st.write("**Toxicity Score:**", result["toxicity_score"])
        st.write("**Sentiment:**", result["sentiment"])

        highlighted = highlight_text(text, result["targets"])
        st.write("**Highlighted Text:**", highlighted)

        st.write("**Explanation:**", result["explanation"])


# ---------------- NEWS ----------------
st.subheader("🌍 Live News Scan")

articles = get_news()

for a in articles:
    st.write("## 🧾 News")
    st.write("###", a["title"])

    if a["summary"] != "No summary available":
        st.write(a["summary"])

    result = analyze_text(a["title"])

    st.write("### 🔍 AI Analysis")
    st.write("**Category:**", result["category"])
    st.write("**Score:**", result["toxicity_score"])
    st.write("**Explanation:**", result["explanation"])

    st.write("[Read full article]", a["link"])
    st.divider()
