from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# ---------------- MODELS ----------------
toxicity_model = pipeline("text-classification", model="unitary/toxic-bert")
sentiment_model = pipeline("sentiment-analysis")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- BIAS KNOWLEDGE BASE ----------------
bias_db = {
    "gender": [
        "women are not good leaders",
        "men are better engineers",
        "women belong in the kitchen"
    ],
    "race": [
        "some races are superior",
        "certain ethnic groups are less intelligent"
    ],
    "religion": [
        "this religion is violent",
        "people of that faith are dangerous"
    ]
}

bias_texts = []
bias_labels = []

for label, items in bias_db.items():
    for i in items:
        bias_texts.append(i)
        bias_labels.append(label)

bias_embeddings = embedder.encode(bias_texts, convert_to_tensor=True)


# ---------------- SEMANTIC MATCH ENGINE ----------------
def semantic_match(text):
    vec = embedder.encode(text, convert_to_tensor=True)
    scores = util.cos_sim(vec, bias_embeddings)[0]

    top_idx = int(scores.argmax())

    return {
        "score": float(scores[top_idx]),
        "label": bias_labels[top_idx],
        "matched_text": bias_texts[top_idx],
        "all_scores": scores.tolist()
    }


# ---------------- MAIN ANALYSIS ----------------
def analyze_text(text):
    text_l = text.lower()

    tox = toxicity_model(text)[0]
    sent = sentiment_model(text)[0]

    toxicity = float(tox["score"])
    bias = semantic_match(text)

    violence_words = ["kill", "murder", "attack", "bomb", "war"]
    news_words = ["said", "reported", "according", "according to", "bbc", "cnn"]

    violence = sum(w in text_l for w in violence_words) / 3
    news = sum(w in text_l for w in news_words) / 3

    return {
        "text": text,
        "toxicity": toxicity,
        "sentiment": sent["label"],
        "bias_score": bias["score"],
        "bias_type": bias["label"],
        "bias_match": bias["matched_text"],
        "violence_score": violence,
        "news_score": news
    }
