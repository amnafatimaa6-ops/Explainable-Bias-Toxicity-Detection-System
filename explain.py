def explain(result, text):
    reasons = []
    t = text.lower()

    if result["bias_score"] > 0.65:
        reasons.append(f"High semantic similarity to {result['bias_type']} bias examples")

    if result["toxicity"] > 0.7:
        reasons.append("High toxicity detected by transformer model")

    if any(w in t for w in ["women", "men", "race", "religion"]):
        reasons.append("Sensitive demographic terms detected")

    if len(reasons) == 0:
        reasons.append("No strong harmful or biased semantic patterns detected")

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
