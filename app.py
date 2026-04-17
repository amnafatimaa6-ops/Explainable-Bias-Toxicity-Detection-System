import streamlit as st
from model import analyze_text, highlight_text
from news import get_news

st.set_page_config(page_title="AI Ethics Radar", layout="centered")

st.title("🧠 AI Ethics Radar — Live Bias Scanner")

# ----------------------------
# USER INPUT
# ----------------------------
st.subheader("🔍 Analyze Text")

text = st.text_area("Enter text")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter text")
    else:
        result = analyze_text(text)

        st.write("### 🧠 AI Analysis")
        st.write("**Risk Level:**", result["risk_level"])
        st.write("**Toxicity Score:**", result["toxicity"])

        highlighted = highlight_text(text, result["flagged_words"])
        st.write("**Highlighted Text:**", highlighted)

        st.write("**Explanation:**", result["explanation"])


# ----------------------------
# LIVE NEWS SECTION
# ----------------------------
st.subheader("🌍 Live News Scan")

articles = get_news()

for a in articles:
    st.write("## 🧾 News")

    st.write("###", a["title"])
    st.write(a["summary"])

    result = analyze_text(a["title"])

    st.write("### 🔍 AI Analysis")
    st.json(result)

    st.write("[Read full article]", a["link"])
    st.divider()
