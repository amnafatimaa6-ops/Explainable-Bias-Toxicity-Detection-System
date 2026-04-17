from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

toxicity_model = pipeline("text-classification", model="unitary/toxic-bert")
sentiment_model = pipeline("sentiment-analysis")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- BIAS DATABASE ----------------
bias_db = {
    "gender": [
        "women are not good leaders",
        "men are better engineers",
        "women belong in the kitchen"
    ],
    "race": [
        "some races are superior",
        "certain ethnic groups are less intelligent"
    ]
}

bias_texts = []
bias_labels = []

for label, items in bias_db.items():
    for item in items:
        bias_texts.append(item)
        bias_labels.append(label)

bias_embeddings = embedder.encode(bias_texts, convert_to_tensor=True)

# ---------------- INTENT ----------------
def detect_intent(text):
    t = text.lower()

    if "better than" in t or "superior" in t:
        return "comparison"
    if "all" in t or "every" in t:
        return "generalization"
    return "statement"


def amplify_bias(score, text):
    intent = detect_intent(text)

    if intent == "generalization":
        score *= 1.4
    elif intent == "comparison":
        score *= 1.3

    return min(score, 1.0)


# ---------------- BIAS MATCH ----------------
def semantic_bias(text):
    vec = embedder.encode(text, convert_to_tensor=True)
    scores = util.cos_sim(vec, bias_embeddings)[0]

    idx = int(scores.argmax())

    return {
        "score": float(scores[idx]),
        "label": bias_labels[idx],
        "match": bias_texts[idx]
    }


# ---------------- MAIN ----------------
def analyze_text(text):
    tox = toxicity_model(text)[0]
    sent = sentiment_model(text)[0]

    raw_bias = semantic_bias(text)
    bias_score = amplify_bias(raw_bias["score"], text)

    return {
        "text": text,
        "toxicity": float(tox["score"]),
        "sentiment": sent["label"],
        "bias_score": bias_score,
        "bias_type": raw_bias["label"],
        "bias_match": raw_bias["match"]
    }
