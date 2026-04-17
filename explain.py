def explain_result(result):
    reasons = []

    if result["bias_score"] > 0.65:
        reasons.append("High semantic similarity to known bias patterns")

    if result["toxicity"] > 0.7:
        reasons.append("High toxicity confidence from transformer model")

    if result["violence_score"] > 0.3:
        reasons.append("Presence of violence-related keywords")

    if result["news_score"] > 0.4:
        reasons.append("Detected journalistic/reporting structure")

    if not reasons:
        reasons.append("No strong harmful signals detected")

    return " | ".join(reasons)
