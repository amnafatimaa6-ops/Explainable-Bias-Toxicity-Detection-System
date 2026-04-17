def explain(result, text):
    reasons = []
    t = text.lower()

    if result["bias_score"] > 0.6:
        reasons.append(
            f"Semantic match with biased example: '{result['bias_match']}'"
        )

    if "better than" in t:
        reasons.append("Detected comparison between groups (strong stereotype pattern)")

    if "all" in t or "every" in t:
        reasons.append("Detected generalization across a group")

    if result["toxicity"] > 0.7:
        reasons.append("High toxicity detected from transformer model")

    if len(reasons) == 0:
        reasons.append("No strong harmful semantic or emotional patterns detected")

    return " | ".join(reasons)


def risk_level(score):
    if score > 0.75:
        return "🔴 High"
    elif score > 0.5:
        return "🟠 Medium"
    elif score > 0.25:
        return "🟡 Low"
    else:
        return "🟢 None"
