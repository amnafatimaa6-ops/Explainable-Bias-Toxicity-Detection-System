from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# ---------------- MODELS ----------------
toxicity_model = pipeline("text-classification", model="unitary/toxic-bert")
sentiment_model = pipeline("sentiment-analysis")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- BIAS KNOWLEDGE BASE ----------------
bias_database = {
    "gender": [
        "women are not good leaders",
        "men are better engineers",
        "women are less capable in leadership"
    ],
    "race": [
        "some races are superior to others",
        "certain ethnic groups are less intelligent"
    ],
    "religion": [
        "this religion is violent",
        "people of that faith are dangerous"
    ],
    "nationality": [
        "immigrants are dangerous",
        "foreigners steal jobs"
    ]
}

# flatten embeddings
bias_texts = []
bias_labels = []

for label, examples in bias_database.items():
    for ex in examples:
        bias_texts.append(ex)
        bias_labels.append(label)

bias_embeddings = embedder.encode(bias_texts, convert_to_tensor=True)

# ---------------- SEMANTIC ENGINE ----------------
def semantic_match(text):
    vec = embedder.encode(text, convert_to_tensor=True)
    scores = util.cos_sim(vec, bias_embeddings)[0]

    best_idx = int(scores.argmax())
    return float(scores[best_idx]), bias_labels[best_idx]


# ---------------- MAIN FUNCTION ----------------
def analyze_text(text):
    text_lower = text.lower()

    tox = toxicity_model(text)[0]
    sentiment = sentiment_model(text)[0]

    tox_score = float(tox["score"])

    # semantic bias detection
    bias_score, bias_type = semantic_match(text)

    # simple violence/news heuristics (still useful)
    violence_keywords = ["killed", "murder", "attack", "bomb", "war", "shooting"]
    news_keywords = ["said", "reported", "according to", "bbc", "cnn"]

    violence_score = sum(k in text_lower for k in violence_keywords) / 3
    news_score = sum(k in text_lower for k in news_keywords) / 3

    # ---------------- DECISION ENGINE ----------------

    if bias_score > 0.65:
        category = f"Bias ({bias_type})"
        explanation = f"Semantically similar to {bias_type}-related biased statements."

    elif violence_score > 0.3:
        category = "Violence / Crime Context"
        explanation = "Contains references to harmful or violent events."

    elif news_score > 0.4:
        category = "News / Reporting"
        explanation = "Factual or informational reporting content."

    elif tox_score > 0.7:
        category = "Toxic Language"
        explanation = "Aggressive or harmful tone detected."

    elif sentiment["label"] == "NEGATIVE":
        category = "Negative Opinion"
        explanation = "Negative sentiment without bias or toxicity."

    else:
        category = "Neutral"
        explanation = "No harmful semantic patterns detected."

    return {
        "category": category,
        "bias_score": round(bias_score, 3),
        "bias_type": bias_type,
        "toxicity": round(tox_score, 3),
        "violence_score": round(violence_score, 3),
        "news_score": round(news_score, 3),
        "sentiment": sentiment["label"],
        "explanation": explanation
    }
