import streamlit as st
import requests
import feedparser
from model import BiasModel

st.set_page_config(page_title="AI Ethics Radar Pro", layout="wide")

st.title("🧠 AI Ethics Radar Pro")
st.write("Real-time Bias + Toxicity + Media Intelligence System")

# -------------------------
# SAFE MODEL LOADING
# -------------------------
@st.cache_resource
def load_model():
    m = BiasModel()
    m.train()
    return m

model = load_model()

# -------------------------
# DATA SOURCES
# -------------------------
def get_reddit():
    try:
        url = "https://www.reddit.com/r/news/.json"
        headers = {"User-agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json()
        return [x["data"]["title"] for x in data["data"]["children"][:5]]
    except:
        return []

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

    result = model.predict(text)

    st.subheader("AI Output")

    st.metric("Toxicity", f"{result.get('toxicity', 0):.2f}")
    st.metric("Bias", f"{result.get('bias_signal', 0):.2f}")
    st.metric("Risk", f"{result.get('risk_layer', 0):.2f}")

    st.subheader("Explainability")

    for word, score in model.explain(text):
        st.write(f"{word} → {score:.3f}")

    if result.get("toxicity", 0) > 0.6:
        st.error("⚠ Toxic content detected")

    if result.get("bias_signal", 0) > 0.6:
        st.warning("⚠ Bias detected")

# -------------------------
# LIVE FEED
# -------------------------
st.header("🌍 Live Intelligence Feed")

if st.button("Run Live Scan"):

    items = [("Reddit", x) for x in get_reddit()] + [("News", x) for x in get_news()]

    for source, text in items:

        result = model.predict(text)

        st.markdown(f"### 🧾 {source}")
        st.write(text)

        st.metric("Toxicity", f"{result.get('toxicity', 0):.2f}")
        st.metric("Bias", f"{result.get('bias_signal', 0):.2f}")
        st.metric("Risk", f"{result.get('risk_layer', 0):.2f}")

        if result.get("risk_layer", 0) > 0.6:
            st.warning("⚠ High contextual risk detected")

        st.markdown("---")
