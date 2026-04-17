import streamlit as st
import requests
import feedparser
from model import BiasModel

st.set_page_config(page_title="AI Ethics Radar", layout="wide")

st.title("🧠 AI Ethics Radar (Live Intelligence System)")

st.write("Detect bias, toxicity, and framing differences in real-world content")

# ------------------------
# MODEL (CACHED)
# ------------------------
@st.cache_resource
def load_model():
    m = BiasModel()
    m.train()
    return m

model = load_model()

# ------------------------
# REDDIT SCRAPER (NO API)
# ------------------------
def get_reddit():
    try:
        url = "https://www.reddit.com/r/news/.json"
        headers = {"User-agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json()

        return [item["data"]["title"] for item in data["data"]["children"][:5]]
    except:
        return []

# ------------------------
# NEWS RSS
# ------------------------
def get_news():
    try:
        feed = feedparser.parse("https://news.google.com/rss")
        return [e.title for e in feed.entries[:5]]
    except:
        return []

# ------------------------
# TEXT ANALYSIS
# ------------------------
st.header("🔍 Analyze Custom Text")

text = st.text_area("Enter text")

if st.button("Analyze") and text:

    result = model.predict(text)

    st.subheader("AI Prediction")

    st.write(result)

    st.subheader("Explainability")

    for word, score in model.explain(text):
        st.write(f"{word} → {score:.3f}")

    # ⚡ threshold logic (IMPORTANT)
    if result["toxicity"] > 0.6:
        st.error("⚠ High Toxicity Detected")

    if result["bias_signal"] > 0.6:
        st.warning("⚠ High Bias Signal Detected")

# ------------------------
# LIVE INTELLIGENCE FEED
# ------------------------
st.header("🌍 Live Intelligence Feed")

if st.button("Run Live Scan"):

    reddit = get_reddit()
    news = get_news()

    items = [("Reddit", x) for x in reddit] + [("News", x) for x in news]

    for source, text in items:

        result = model.predict(text)

        st.markdown(f"### 🧾 {source}")
        st.write(text)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Toxicity Score", f"{result['toxicity']:.2f}")

        with col2:
            st.metric("Bias Score", f"{result['bias_signal']:.2f}")

        # ⚡ intelligence rules (your “wow factor”)
        if result["bias_signal"] > 0.7:
            st.error("⚠ Potential framing bias detected")

        if result["toxicity"] > 0.7:
            st.error("⚠ Toxic language detected")

        st.markdown("---")
