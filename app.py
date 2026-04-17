import streamlit as st
import requests
import feedparser
from model import BiasModel

st.set_page_config(page_title="AI Ethics Radar Pro", layout="wide")

st.markdown(
    """
    <h1 style='text-align:center; color:#00ffcc;'>🧠 AI Ethics Radar Pro</h1>
    <p style='text-align:center; color:gray;'>Bias • Toxicity • Semantic Intelligence System</p>
    """,
    unsafe_allow_html=True
)

# -------------------------
# MODEL LOAD
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
        return [x["data"]["title"] for x in data["data"]["children"][:6]]
    except:
        return []

def get_news():
    try:
        feed = feedparser.parse("https://news.google.com/rss")
        return [x.title for x in feed.entries[:6]]
    except:
        return []

# -------------------------
# INPUT SECTION
# -------------------------
st.header("🔍 Analyze Text")

text = st.text_area("Enter text")

if st.button("Analyze") and text:

    result = model.predict(text)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Toxicity", f"{result.get('toxicity', 0):.2f}")
    col2.metric("Bias", f"{result.get('bias_signal', 0):.2f}")
    col3.metric("Risk", f"{result.get('risk_layer', 0):.2f}")
    col4.metric("Semantic", f"{result.get('semantic', 0):.2f}")

    st.subheader("Explainability Layer")

    for word, score in model.explain(text):
        color = "red" if score < 0 else "green"
        st.markdown(f"<span style='color:{color}; font-size:18px'>{word}</span>", unsafe_allow_html=True)

    if result.get("bias_signal", 0) > 0.6:
        st.error("⚠ Bias detected")

    if result.get("toxicity", 0) > 0.6:
        st.error("⚠ Toxicity detected")

# -------------------------
# LIVE FEED
# -------------------------
st.markdown("---")
st.header("🌍 Live Intelligence Feed")

if st.button("Run Live Scan"):

    items = [("Reddit", x) for x in get_reddit()] + [("News", x) for x in get_news()]

    for source, item in items:

        result = model.predict(item)

        st.markdown(f"### 🧾 {source}")
        st.write(item)

        col1, col2, col3 = st.columns(3)

        col1.metric("Toxicity", f"{result.get('toxicity', 0):.2f}")
        col2.metric("Bias", f"{result.get('bias_signal', 0):.2f}")
        col3.metric("Risk", f"{result.get('risk_layer', 0):.2f}")

        if result.get("risk_layer", 0) > 0.6:
            st.warning("⚠ High contextual risk detected")

        st.markdown("---")
