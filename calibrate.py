import numpy as np

def calibrate(score):
    # smooth probability curve (realistic risk scaling)
    return float(1 / (1 + np.exp(-10 * (score - 0.5))))


def risk_level(score):
    if score > 0.8:
        return "🔴 Critical"
    elif score > 0.6:
        return "🟠 High"
    elif score > 0.35:
        return "🟡 Medium"
    else:
        return "🟢 Low"
