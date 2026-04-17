import streamlit as st
import requests
import feedparser
from model import BiasModel

st.set_page_config(
    page_title="AI Ethics Radar Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# HEADER (BOSS LEVEL LOOK)
# -----------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #00ffcc;'>
    🧠 AI Ethics Radar Pro
    </h1>
    <h4 style='text-align: center; color: gray;'>
    Real-time Bias • Toxicity • Media Framing Intelligence System
    </h4>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    m = BiasModel()
    m.train()
    return m

model = load_model()

# -----------------------------
# SIDEBAR (CONTROL PANEL)
# -----------------------------
st.sidebar.title("⚙ Control Panel")
threshold = st.sidebar.slider("Risk Sensitivity", 0.0, 1.0, 0.6)

st.sidebar.markdown("---")
st.sidebar.info("Live system using Reddit + News RSS feeds")

# -----------------------------
# SOURCES
# -----------------------------
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

# -----------------------------
# TEXT INPUT
# -----------------------------
st.markdown("## 🔍 Analyze Text")

text = st.text_area("Enter text for AI ethics analysis")

# -----------------------------
# COLOR RISK FUNCTION
# -----------------------------
def risk_color(score):
    if score > 0.7:
        return "red"
    elif score > 0.4:
        return "orange"
    return "green"

# -----------------------------
# ANALYSIS SECTION
# -----------------------------
if st.button("Run Analysis") and text:

    result = model.predict(text)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🔥 Toxicity")
        st.metric("Score", f"{result['toxicity']:.2f}")
        st.progress(min(int(result["toxicity"] * 100), 100))

    with col2:
        st.markdown("### ⚖ Bias")
        st.metric("Score", f"{result['bias_signal']:.2f}")
        st.progress(min(int(result["bias_signal"] * 100), 100))

    with col3:
        st.markdown("### 🧠 Risk Engine")
        st.metric("Score", f"{result['risk_layer']:.2f}")
        st.progress(min(int(result["risk_layer"] * 100), 100))

    # -----------------------------
    # ALERT SYSTEM
    # -----------------------------
    if result["bias_signal"] > threshold:
        st.error("⚠ High Bias Detected")

    if result["toxicity"] > threshold:
        st.error("⚠ High Toxicity Detected")

    # -----------------------------
    # EXPLAINABILITY PANEL
    # -----------------------------
    st.markdown("## 🧾 Explainability Layer")

    explanation = model.explain(text)

    for word, score in explanation:
        color = risk_color(abs(score))
        st.markdown(f"<span style='color:{color}; font-size:18px;'>{word}</span>", unsafe_allow_html=True)

# -----------------------------
# LIVE INTELLIGENCE FEED
# -----------------------------
st.markdown("---")
st.markdown("## 🌍 Live Intelligence Feed")

if st.button("Run Live Scan"):

    items = [("Reddit", x) for x in get_reddit()] + [("News", x) for x in get_news()]

    for source, item in items:

        result = model.predict(item)

        st.markdown(
            f"""
            <div style="
                background-color:#111;
                padding:15px;
                border-radius:12px;
                margin-bottom:10px;
                border:1px solid #333;
            ">
                <h4 style="color:#00ffcc;">🧾 {source}</h4>
                <p style="color:white;">{item}</p>

                <p style="color:orange;">Toxicity: {result['toxicity']:.2f}</p>
                <p style="color:cyan;">Bias: {result['bias_signal']:.2f}</p>
                <p style="color:gray;">Risk: {result['risk_layer']:.2f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if result["risk_layer"] > threshold:
            st.warning("⚠ Elevated Risk Detected")
