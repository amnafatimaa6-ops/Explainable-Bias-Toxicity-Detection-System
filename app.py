import streamlit as st
import feedparser
from model import EthicsRadarSafe

st.set_page_config(page_title="AI Ethics Radar", layout="wide")

st.title("🧠 AI Ethics Radar — Deployment Safe Version")

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    m = EthicsRadarSafe()
    m.train()
    return m

model = load_model()

# -------------------------
# INPUT
# -------------------------
st.header("🔍 Analyze Text")

text = st.text_area("Enter text")

if st.button("Analyze") and text:

    r = model.predict(text)

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Toxicity", f"{r['toxicity']:.2f}")
    c2.metric("Bias", f"{r['bias']:.2f}")
    c3.metric("Risk", f"{r['risk']:.2f}")
    c4.metric("Framing", f"{r['framing']:.2f}")
    c5.metric("Sentiment", f"{r['sentiment']:.2f}")

# -------------------------
# LIVE FEED
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
