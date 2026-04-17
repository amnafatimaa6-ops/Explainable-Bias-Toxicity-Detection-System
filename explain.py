\def explain(result, text):
    reasons = []
    t = text.lower()

    if result["bias_score"] > 0.6:
        reasons.append(f"Matched biased pattern: '{result['bias_match']}'")

    if "better than" in t:
        reasons.append("Detected comparison between groups")

    if "all" in t or "every" in t:
        reasons.append("Detected generalization over a group")

    if result["toxicity"] > 0.7:
        reasons.append("High toxicity detected")

    if not reasons:
        reasons.append("No strong harmful semantic patterns detected")

    return " | ".join(reasons)


def risk_level(score):
    if score > 0.75:
        return "🔴 High"
    elif score > 0.5:
        return "🟠 Medium"
    elif score > 0.25:
        return "🟡 Low"
    return "🟢 None"
