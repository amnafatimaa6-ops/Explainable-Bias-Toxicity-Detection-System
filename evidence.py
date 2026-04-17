def build_evidence(result):
    evidence = []

    if result["bias_score"] > 0.6:
        evidence.append({
            "type": "semantic_bias_match",
            "matched_example": result["bias_match"],
            "reason": "Input is semantically similar to known biased statement"
        })

    if result["toxicity"] > 0.7:
        evidence.append({
            "type": "toxicity_model",
            "reason": "High confidence toxic classification from transformer model"
        })

    if result["violence_score"] > 0.3:
        evidence.append({
            "type": "keyword_signal",
            "reason": "Violence-related lexical indicators detected"
        })

    return evidence
