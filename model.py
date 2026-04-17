import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

class BiasModelV2:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=6000, stop_words="english")
        self.models = {}
        self.sia = SentimentIntensityAnalyzer()

    # -------------------------
    # DATA
    # -------------------------
    def load_data(self):
        data = {
            "text": [
                "I love this idea",
                "This is terrible and disgusting",
                "Government supports education reform",
                "This group is dangerous and bad",
                "Women are not good leaders",
                "People are kind and helpful",
                "He is a stupid and useless person",
                "Education improves society",
                "War and violence are increasing",
                "This is a wonderful achievement",
                "This policy is unfair and biased",
                "Citizens demand justice and equality"
            ],
            "toxicity": [0,1,0,1,1,0,1,0,1,0,1,0],
            "identity_attack": [0,0,0,1,1,0,0,0,0,0,0,0],
            "insult": [0,1,0,1,0,0,1,0,0,0,1,0]
        }

        df = pd.DataFrame(data)
        df["toxicity_label"] = df["toxicity"]
        df["bias_label"] = ((df["identity_attack"]==1) | (df["insult"]==1)).astype(int)
        return df

    # -------------------------
    # TRAIN
    # -------------------------
    def train(self):
        df = self.load_data()
        X = self.vectorizer.fit_transform(df["text"])

        for target in ["toxicity_label", "bias_label"]:
            y = df[target]
            model = LogisticRegression(max_iter=300)
            model.fit(X, y)
            self.models[target] = model

        print("V2 Model trained ✔")

    # -------------------------
    # SENTIMENT LAYER (NEW)
    # -------------------------
    def sentiment_score(self, text):
        score = self.sia.polarity_scores(text)["compound"]
        return abs(score)

    # -------------------------
    # ADVANCED RISK ENGINE (FIXES ZERO OUTPUTS)
    # -------------------------
    def hybrid_risk(self, text):

        text = text.lower()

        strong_bias_words = {
            "women": 0.2,
            "men": 0.1,
            "government": 0.2,
            "leader": 0.3,
            "illegal": 0.5,
            "immigrant": 0.4
        }

        toxicity_words = {
            "hate": 0.7,
            "stupid": 0.6,
            "idiot": 0.6,
            "kill": 0.8,
            "violent": 0.7
        }

        score = 0

        for w,v in strong_bias_words.items():
            if w in text:
                score += v

        for w,v in toxicity_words.items():
            if w in text:
                score += v

        # normalization (IMPORTANT)
        return min(score, 1.0)

    # -------------------------
    # PREDICT (REAL FUSION ENGINE)
    # -------------------------
    def predict(self, text):

        vec = self.vectorizer.transform([text])

        ml_toxic = self.models["toxicity_label"].predict_proba(vec)[0][1]
        ml_bias = self.models["bias_label"].predict_proba(vec)[0][1]

        risk = self.hybrid_risk(text)
        sentiment = self.sentiment_score(text)

        # SMART WEIGHTED FUSION
        toxicity = (ml_toxic*0.45) + (risk*0.35) + (sentiment*0.2)
        bias = (ml_bias*0.5) + (risk*0.4) + (sentiment*0.1)

        return {
            "toxicity": float(round(toxicity,3)),
            "bias": float(round(bias,3)),
            "risk": float(round(risk,3)),
            "sentiment": float(round(sentiment,3))
        }

    # -------------------------
    # EXPLAINABILITY
    # -------------------------
    def explain(self, text):

        vec = self.vectorizer.transform([text])
        features = self.vectorizer.get_feature_names_out()

        model = self.models["toxicity_label"]
        coefs = model.coef_[0]

        indices = vec.nonzero()[1]
        values = vec.data

        scores = []

        for i,v in zip(indices, values):
            scores.append((features[i], v * coefs[i]))

        return sorted(scores, key=lambda x: abs(x[1]), reverse=True)[:8]
