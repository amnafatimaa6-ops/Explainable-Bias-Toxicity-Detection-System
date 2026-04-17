from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# ---------------- ML MODELS ----------------
toxicity_model = pipeline(
    "text-classification",
    model="unitary/toxic-bert"
)

sentiment_model = pipeline("sentiment-analysis")

# ---------------- SEMANTIC MODEL ----------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# reference bias examples (this is your "knowledge base")
bias_examples = [
    "women are not good leaders",
    "men are better at engineering",
    "immigrants are dangerous",
    "people of this group are inferior",
    "certain races are smarter than others"
]

bias_vectors = embedder.encode(bias_examples, convert_to_tensor=True)

# ---------------- CORE FUNCTION ----------------
def semantic_bias_score(text):
    text_vec = embedder.encode(text, convert_to_tensor=True)
    scores = util.cos_sim(text_vec, bias_vectors)
    return float(scores.max())


def analyze_text(text):
    text_lower = text.lower()

    # ML outputs
    tox = toxicity_model(text)[0]
    sentiment = sentiment_model(text)[0]

    tox_score = float(tox["score"])
    sentiment_label = sentiment["label"]

    # semantic bias score (🔥 MAIN UPGRADE)
    bias_score = semantic_bias_score(text)

    # simple violence detection (still useful)
    violence_keywords = ["killed", "murder", "attack", "bomb", "war", "shooting"]
    violence_score = sum(k in text_lower for k in violence_keywords) / 3

    # news detection
    news_keywords = ["said", "reported", "according to", "bbc", "cnn"]
    news_score = sum(k in text_lower for k in news_keywords) / 3

    # ---------------- DECISION ENGINE ----------------

    if bias_score > 0.55:
        category = "Bias / Stereotype"
        explanation = "Semantically similar to known biased statements."

    elif violence_score > 0.3:
        category = "Violence / Crime Context"
        explanation = "Mentions violent or harmful real-world events."

    elif news_score > 0.4:
        category = "News / Reporting"
        explanation = "Informational reporting content."

    elif tox_score > 0.7:
        category = "Toxic Language"
        explanation = "Aggressive or harmful tone detected."

    elif sentiment_label == "NEGATIVE":
        category = "Negative Opinion"
        explanation = "Negative sentiment but not necessarily bias."

    else:
        category = "Neutral"
        explanation = "No harmful or biased meaning detected."

    return {
        "category": category,
        "toxicity": round(tox_score, 3),
        "bias_score": round(bias_score, 3),
        "violence_score": round(violence_score, 3),
        "news_score": round(news_score, 3),
        "sentiment": sentiment_label,
        "explanation": explanation
    }


def highlight_text(text):
    return text
