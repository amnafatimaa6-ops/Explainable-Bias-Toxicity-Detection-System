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

bias_texts = []
bias_labels = []

for label, examples in bias_database.items():
    for ex in examples:
        bias_texts.append(ex)
        bias_labels.append(label)

bias_embeddings = embedder.encode(bias_texts, convert_to_tensor=True)


# ---------------- SEMANTIC BIAS ----------------
def semantic_bias(text):
    vec = embedder.encode(text, convert_to_tensor=True)
    scores = util.cos_sim(vec, bias_embeddings)[0]

    best_idx = int(scores.argmax())
    return float(scores[best_idx]), bias_labels[best_idx]


# ---------------- MAIN ANALYSIS ----------------
def analyze_text(text):
    text_lower = text.lower()

    tox = toxicity_model(text)[0]
    sentiment = sentiment_model(text)[0]

    toxicity = float(tox["score"])

    bias_score, bias_type = semantic_bias(text)

    violence_words = ["killed", "murder", "attack", "bomb", "war", "shooting"]
    news_words = ["said", "reported", "according to", "statement", "bbc", "cnn"]

    violence_score = sum(w in text_lower for w in violence_words) / 3
    news_score = sum(w in text_lower for w in news_words) / 3

    # ---------------- DECISION ENGINE ----------------

    if bias_score > 0.65:
        category = f"Bias ({bias_type})"
    elif violence_score > 0.3:
        category = "Violence / Crime Context"
    elif news_score > 0.4:
        category = "News / Reporting"
    elif toxicity > 0.7:
        category = "Toxic Language"
    elif sentiment["label"] == "NEGATIVE":
        category = "Negative Opinion"
    else:
        category = "Neutral"

    return {
        "category": category,
        "bias_score": round(bias_score, 3),
        "bias_type": bias_type,
        "toxicity": round(toxicity, 3),
        "violence_score": round(violence_score, 3),
        "news_score": round(news_score, 3),
        "sentiment": sentiment["label"]
    }
