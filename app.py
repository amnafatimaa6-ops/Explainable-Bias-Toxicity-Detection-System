import streamlit as st
import feedparser
from model import EthicsRadarStable

st.set_page_config(page_title="AI Ethics Radar", layout="wide")

st.title("🧠 AI Ethics Radar — Stable Research System")

# -------------------------
# LOAD MODEL (SAFE)
# -------------------------
model = EthicsRadarStable()

with st.spinner("Training AI model..."):
    model.train()

# -------------------------
# INPUT SECTION
# -------------------------
st.header("🔍 Analyze Text")

text = st.text_area("Enter text")

if st.button("Analyze") and text:

    try:
        r = model.predict(text)

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Toxicity", f"{r['toxicity']:.2f}")
        col2.metric("Bias", f"{r['bias']:.2f}")
        col3.metric("Risk", f"{r['risk']:.2f}")
        col4.metric("Framing", f"{r['framing']:.2f}")
        col5.metric("Sentiment", f"{r['sentiment']:.2f}")

    except Exception as e:
        st.error("Prediction error")
        st.exception(e)

# -------------------------
# LIVE NEWS
# -------------------------
st.header("🌍 Live News Scan")

def get_news():
    try:
        feed = feedparser.parse("https://news.google.com/rss")
        return [x.title for x in feed.entries[:5]]
    except:
        return []

if st.button("Run Scan"):

    for item in get_news():

        r = model.predict(item)

        st.markdown("### 🧾 News")
        st.write(item)
        st.json(r)

        st.markdown("---")
