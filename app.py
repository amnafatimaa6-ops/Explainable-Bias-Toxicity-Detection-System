import streamlit as st
import requests
import feedparser
from model import BiasModelV2

st.set_page_config(page_title="AI Ethics Radar V2", layout="wide")

st.markdown("""
<h1 style='text-align:center; color:#00ffd5;'>🧠 AI Ethics Radar V2</h1>
<p style='text-align:center;'>Advanced Bias • Toxicity • Context AI Engine</p>
""", unsafe_allow_html=True)

# -------------------------
# MODEL
# -------------------------
@st.cache_resource
def load_model():
    m = BiasModelV2()
    m.train()
    return m

model = load_model()

# -------------------------
# DATA
# -------------------------
def get_news():
    try:
        feed = feedparser.parse("https://news.google.com/rss")
        return [x.title for x in feed.entries[:5]]
    except:
        return []

# -------------------------
# INPUT
# -------------------------
st.header("🔍 Analyze Text")
text = st.text_area("Enter text")

if st.button("Analyze") and text:

    r = model.predict(text)

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("Toxicity", f"{r['toxicity']:.2f}")
    c2.metric("Bias", f"{r['bias']:.2f}")
    c3.metric("Risk", f"{r['risk']:.2f}")
    c4.metric("Sentiment", f"{r['sentiment']:.2f}")

    st.subheader("Explainability")

    words = model.explain(text)

    for w,s in words:
        color = "red" if s < 0 else "green"
        st.markdown(f"<span style='color:{color}; font-size:18px'>{w}</span>", unsafe_allow_html=True)

    if r["bias"] > 0.6:
        st.error("⚠ Bias detected")

# -------------------------
# LIVE FEED
# -------------------------
st.header("🌍 Live Intelligence Feed")

if st.button("Run Live Scan"):

    for item in get_news():

        r = model.predict(item)

        st.markdown("### 🧾 News")
        st.write(item)

        col1,col2,col3 = st.columns(3)

        col1.metric("Toxicity", f"{r['toxicity']:.2f}")
        col2.metric("Bias", f"{r['bias']:.2f}")
        col3.metric("Risk", f"{r['risk']:.2f}")

        st.markdown("---")
