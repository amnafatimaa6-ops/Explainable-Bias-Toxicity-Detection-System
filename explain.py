def explain(result, text):

    reasons = []

    if result["bias_type"] != "none":
        reasons.append(f"Bias type detected: {result['bias_type']}")

    if result["violence_score"] > 0:
        reasons.append("Violence-related language detected")

    if result["toxicity"] > 0.7:
        reasons.append("High toxicity detected")

    if result["entities"]:
        reasons.append(f"Entities detected: {result['entities']}")

    if not reasons:
        reasons.append("No strong risk patterns detected")

    return " | ".join(reasons)


def risk_level(score):

    if score > 0.75:
        return "🔴 High"
    elif score > 0.5:
        return "🟠 Medium"
    elif score > 0.25:
        return "🟡 Low"
    return "🟢 None"
