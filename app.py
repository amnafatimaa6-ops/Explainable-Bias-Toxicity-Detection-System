import streamlit as st
import requests
import feedparser
from model import BiasModel

st.set_page_config(page_title="AI Ethics Radar", layout="wide")

st.title("🧠 AI Ethics Radar (Live Internet Intelligence System)")

st.write("Detects bias, toxicity, and framing differences across real-world content")

# ------------------------
# LOAD MODEL
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
def get_reddit_posts():
    try:
        url = "https://www.reddit.com/r/news/.json"
        headers = {"User-agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers)
        data = res.json()

        posts = []

        for item in data["data"]["children"][:5]:
            post = item["data"]["title"]
            posts.append(post)

        return posts

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

    st.json(result)

    st.subheader("Explainability")

    for word, score in model.explain(text):
        st.write(f"{word} → {score:.3f}")

# ------------------------
# LIVE INTELLIGENCE FEED
# ------------------------
st.header("🌍 Live Intelligence Feed (Reddit + News)")

if st.button("Run Live Scan"):

    reddit = get_reddit_posts()
    news = get_news()

    all_items = [
        ("Reddit", x) for x in reddit
    ] + [
        ("News", x) for x in news
    ]

    for source, text in all_items:

        result = model.predict(text)

        st.markdown(f"### 🧾 {source}")

        st.write(text)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Toxicity", result["toxicity"])

        with col2:
            st.metric("Bias Signal", result["bias_signal"])

        # 🔥 DOUBLE STANDARD DETECTOR (KEY FEATURE)
        if result["toxicity"] == 1 and "government" in text.lower():
            st.warning("⚠ Possible institutional framing bias detected")

        if result["bias_signal"] == 1:
            st.error("⚠ Bias language detected")

        st.markdown("---")
