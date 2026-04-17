from transformers import pipeline

# ML models
toxicity_model = pipeline(
    "text-classification",
    model="unitary/toxic-bert"
)

sentiment_model = pipeline("sentiment-analysis")

# ---------------- RULE SETS ----------------
bias_patterns = ["women are", "men are", "they are", "all", "always", "never"]
target_groups = ["women", "men", "immigrants", "muslims", "christians", "people"]

violence_keywords = [
    "killed", "murder", "attack", "bomb", "shooting",
    "war", "violence", "assault", "arrest"
]

news_keywords = [
    "said", "reported", "according to", "bbc", "cnn",
    "official", "statement", "announced", "report"
]

# ---------------- HELPERS ----------------
def is_news(text):
    text_lower = text.lower()
    return any(k in text_lower for k in news_keywords)

def analyze_text(text):
    text_lower = text.lower()

    tox = toxicity_model(text)[0]
    sentiment = sentiment_model(text)[0]

    tox_score = float(tox["score"])
    sentiment_label = sentiment["label"]

    # FEATURE SCORES
    bias_score = sum(1 for p in bias_patterns if p in text_lower)
    violence_score = sum(1 for v in violence_keywords if v in text_lower)
    news_score = sum(1 for n in news_keywords if n in text_lower)

    targets = [t for t in target_groups if t in text_lower]

    # NORMALIZE (important)
    bias_score = min(bias_score / 3, 1.0)
    violence_score = min(violence_score / 3, 1.0)
    news_score = min(news_score / 3, 1.0)

    # FINAL DECISION ENGINE
    if bias_score > 0.3 and targets:
        category = "Bias / Stereotype"
        explanation = f"Generalized statement about {', '.join(targets)}."

    elif violence_score > 0.3:
        category = "Violence / Crime Context"
        explanation = "Mentions violence or harmful events."

    elif news_score > 0.4:
        category = "News / Reporting"
        explanation = "Informational or factual reporting content."

    elif tox_score > 0.7:
        category = "Toxic Language"
        explanation = "Aggressive or harmful tone detected."

    elif sentiment_label == "NEGATIVE":
        category = "Negative Opinion"
        explanation = "Negative sentiment but not bias or toxicity."

    else:
        category = "Neutral"
        explanation = "No harmful patterns detected."

    return {
        "category": category,
        "toxicity": round(tox_score, 3),
        "bias_score": round(bias_score, 3),
        "violence_score": round(violence_score, 3),
        "news_score": round(news_score, 3),
        "sentiment": sentiment_label,
        "targets": targets,
        "explanation": explanation
    }


# ---------------- HIGHLIGHTER ----------------
def highlight_text(text, words):
    if not words:
        return text

    for w in words:
        text = text.replace(w, f"**{w}**")

    return text
