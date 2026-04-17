import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from datasets import load_dataset
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")


class EthicsRadarV3:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1,2),
            stop_words="english"
        )

        self.models = {}
        self.sia = SentimentIntensityAnalyzer()

    # -------------------------
    # REAL DATASET (IMPORTANT UPGRADE)
    # -------------------------
    def load_data(self):

        dataset = load_dataset("civil_comments")
        df = pd.DataFrame(dataset["train"]).sample(20000, random_state=42)

        df["toxicity_label"] = (df["toxicity"] > 0.5).astype(int)

        df["bias_label"] = (
            (df["identity_attack"] > 0.5) |
            (df["insult"] > 0.7)
        ).astype(int)

        return df[["text", "toxicity_label", "bias_label"]].dropna()

    # -------------------------
    # TRAINING (CALIBRATED MODEL)
    # -------------------------
    def train(self):

        df = self.load_data()

        X = self.vectorizer.fit_transform(df["text"])

        for target in ["toxicity_label", "bias_label"]:

            y = df[target]

            base = SGDClassifier(loss="log_loss", max_iter=1000)

            model = CalibratedClassifierCV(base, method="isotonic", cv=3)
            model.fit(X, y)

            self.models[target] = model

        print("✔ V3 trained with calibration")

    # -------------------------
    # SENTIMENT LAYER
    # -------------------------
    def sentiment(self, text):

        s = self.sia.polarity_scores(text)
        return abs(s["compound"])

    # -------------------------
    # ADVANCED FRAMING DETECTOR
    # -------------------------
    def framing_bias(self, text):

        text = text.lower()

        patterns = [
            "are bad", "are evil", "are dangerous",
            "all", "always", "never", "most",
            "typical of", "those people"
        ]

        score = sum(0.2 for p in patterns if p in text)

        return min(score, 1.0)

    # -------------------------
    # DOMAIN RISK ENGINE (IMPROVED)
    # -------------------------
    def risk_engine(self, text):

        text = text.lower()

        lexicon = {
            "violence": 0.5,
            "war": 0.4,
            "hate": 0.6,
            "kill": 0.7,
            "attack": 0.5,
            "racist": 0.9,
            "crime": 0.3,
            "women": 0.15,
            "government": 0.1
        }

        score = 0
        for k,v in lexicon.items():
            if k in text:
                score += v

        return min(score, 1.0)

    # -------------------------
    # NORMALIZATION (IMPORTANT FOR RESEARCH STABILITY)
    # -------------------------
    def normalize(self, x):
        return float(np.clip(x, 0.02, 0.98))

    # -------------------------
    # FINAL PREDICTION (RESEARCH FORMULA)
    # -------------------------
    def predict(self, text):

        X = self.vectorizer.transform([text])

        ml_toxic = self.models["toxicity_label"].predict_proba(X)[0][1]
        ml_bias = self.models["bias_label"].predict_proba(X)[0][1]

        risk = self.risk_engine(text)
        sentiment = self.sentiment(text)
        framing = self.framing_bias(text)

        # LOGIT-STABLE FUSION (research style weighting)
        toxicity = (
            0.40 * ml_toxic +
            0.25 * risk +
            0.20 * framing +
            0.15 * sentiment
        )

        bias = (
            0.40 * ml_bias +
            0.30 * framing +
            0.20 * risk +
            0.10 * sentiment
        )

        return {
            "toxicity": self.normalize(toxicity),
            "bias": self.normalize(bias),
            "risk": self.normalize(risk),
            "framing": self.normalize(framing),
            "sentiment": self.normalize(sentiment)
        }
