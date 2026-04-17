import streamlit as st
import traceback
import feedparser
from model import EthicsRadarStable

st.set_page_config(page_title="AI Ethics Radar", layout="wide")

st.title("🧠 AI Ethics Radar — Debug Mode")

# -------------------------
# MODEL LOAD
# -------------------------
try:
    model = EthicsRadarStable()
    model.train()
except Exception as e:
    st.error("❌ MODEL TRAINING FAILED")
    st.code(traceback.format_exc())
    st.stop()

# -------------------------
# INPUT
# -------------------------
st.header("🔍 Analyze Text")
text = st.text_area("Enter text")

if st.button("Analyze") and text:

    try:
        r = model.predict(text)

        st.json(r)

    except Exception as e:
        st.error("❌ PREDICTION FAILED")
        st.code(traceback.format_exc())

# -------------------------
# NEWS
# -------------------------
st.header("🌍 Live Feed")

def get_news():
    try:
        return [x.title for x in feedparser.parse("https://news.google.com/rss").entries[:5]]
    except Exception as e:
        st.error("RSS ERROR")
        st.code(traceback.format_exc())
        return []

if st.button("Run Scan"):

    for item in get_news():
        try:
            r = model.predict(item)
            st.write(item)
            st.json(r)

        except Exception as e:
            st.error("NEWS PREDICTION FAILED")
            st.code(traceback.format_exc())
