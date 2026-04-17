import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class BiasModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        self.models = {}

    # -------------------------
    # TRAINING DATA
    # -------------------------
    def load_data(self):
        data = {
            "text": [
                "I love this idea, it's amazing",
                "This is terrible and disgusting",
                "Government supports education reform",
                "This group is dangerous and bad",
                "Women are not good leaders",
                "People are kind and helpful",
                "He is a stupid and useless person",
                "Education improves society",
                "War and violence are increasing",
                "This is a wonderful achievement"
            ],
            "toxicity": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
            "identity_attack": [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            "insult": [0, 1, 0, 1, 0, 0, 1, 0, 0, 0]
        }

        df = pd.DataFrame(data)

        df["toxicity_label"] = df["toxicity"]
        df["bias_label"] = (
            (df["identity_attack"] == 1) | (df["insult"] == 1)
        ).astype(int)

        return df

    # -------------------------
    # TRAIN MODEL
    # -------------------------
    def train(self):
        df = self.load_data()

        X = self.vectorizer.fit_transform(df["text"])

        for target in ["toxicity_label", "bias_label"]:
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = LogisticRegression(max_iter=300)
            model.fit(X_train, y_train)

            self.models[target] = model

        print("Model trained ✔")

    # -------------------------
    # RISK ENGINE
    # -------------------------
    def hybrid_risk_score(self, text):
        text = text.lower()

        risk_words = {
            "war": 0.4,
            "violence": 0.5,
            "kill": 0.7,
            "murder": 0.8,
            "attack": 0.5,
            "hate": 0.7,
            "racist": 0.9,
            "conflict": 0.3,
            "women": 0.2,
            "men": 0.1,
            "government": 0.1
        }

        score = 0
        for w, val in risk_words.items():
            if w in text:
                score += val

        return min(score, 1.0)

    # -------------------------
    # PREDICTION (SAFE + STABLE)
    # -------------------------
    def predict(self, text):
        vec = self.vectorizer.transform([text])

        ml_toxic = self.models["toxicity_label"].predict_proba(vec)[0][1]
        ml_bias = self.models["bias_label"].predict_proba(vec)[0][1]

        risk = self.hybrid_risk_score(text)

        final_toxic = (ml_toxic * 0.6) + (risk * 0.4)
        final_bias = (ml_bias * 0.6) + (risk * 0.4)

        return {
            "toxicity": round(float(final_toxic), 3),
            "bias_signal": round(float(final_bias), 3),
            "ml_toxicity": round(float(ml_toxic), 3),
            "risk_layer": round(float(risk), 3)
        }

    # -------------------------
    # EXPLAINABILITY
    # -------------------------
    def explain(self, text):
        vec = self.vectorizer.transform([text])

        feature_names = self.vectorizer.get_feature_names_out()
        indices = vec.nonzero()[1]
        values = vec.data

        model = self.models["toxicity_label"]
        coefs = model.coef_[0]

        scores = []

        for idx, tfidf_val in zip(indices, values):
            word = feature_names[idx]
            score = tfidf_val * coefs[idx]
            scores.append((word, score))

        return sorted(scores, key=lambda x: abs(x[1]), reverse=True)[:10]
