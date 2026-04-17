import streamlit as st
import feedparser
from model import EthicsRadarStable

st.set_page_config(page_title="Ethics Radar Stable", layout="wide")

st.title("🧠 AI Ethics Radar — Stable Version")

# IMPORTANT FIX: no fragile cache first run
model = EthicsRadarStable()
model.train()

st.header("🔍 Text Analysis")

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
        st.error(f"Model error: {e}")

# -------------------------
# LIVE FEED SAFE MODE
# -------------------------
st.header("🌍 Live Feed")

def news():
    try:
        feed = feedparser.parse("https://news.google.com/rss")
        return [x.title for x in feed.entries[:5]]
    except:
        return []

if st.button("Run Live Scan"):

    for item in news():

        r = model.predict(item)

        st.markdown("### 🧾 News")
        st.write(item)
        st.json(r)

        st.markdown("---")
