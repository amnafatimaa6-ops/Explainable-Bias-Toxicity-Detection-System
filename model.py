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

    # feature detection
    generalization = any(p in text_lower for p in bias_patterns)
    targets = [t for t in target_groups if t in text_lower]
    violence = any(v in text_lower for v in violence_keywords)
    news_flag = is_news(text_lower)

    # ---------------- SMART CLASSIFICATION ----------------

    if news_flag:
        category = "News / Factual Reporting"
        explanation = "This is informational reporting and not personal opinion."

    elif generalization and targets:
        category = "Bias / Stereotype"
        explanation = f"This statement generalizes about {', '.join(targets)}."

    elif violence:
        category = "Violence / Crime Context"
        explanation = "This text refers to violent or criminal events."

    elif tox_score > 0.7:
        category = "Toxic Language"
        explanation = "Aggressive or harmful tone detected."

    elif sentiment_label == "NEGATIVE":
        category = "Negative Opinion"
        explanation = "Negative sentiment but not necessarily biased."

    else:
        category = "Neutral"
        explanation = "No harmful or biased patterns detected."

    return {
        "category": category,
        "toxicity_score": round(tox_score, 3),
        "sentiment": sentiment_label,
        "targets": targets,
        "generalization": generalization,
        "violence": violence,
        "news": news_flag,
        "explanation": explanation
    }


# ---------------- HIGHLIGHTER ----------------
def highlight_text(text, words):
    if not words:
        return text

    for w in words:
        text = text.replace(w, f"**{w}**")

    return text
