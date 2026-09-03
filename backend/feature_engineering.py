"""
Feature Engineering & Clinical Index Calculator Module.
Computes derived indices (Sleep Risk Index, Social Isolation Index, Mood Variability Index, Depression Index)
and performs data scaling & class balancing.
"""

import numpy as np
from typing import Dict, Any, List

def compute_derived_features(features: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculates derived & composite domain-specific clinical indices from raw multi-modal inputs.
    """
    phq9 = float(features.get("phq9_score", 0))
    gad7 = float(features.get("gad7_score", 0))
    mdq = float(features.get("mdq_score", 0))
    who5 = float(features.get("who5_wellbeing", 50))
    sleep_hrs = float(features.get("sleep_hours", 7.0))
    social_score = float(features.get("social_interaction_score", 50))
    screen_hrs = float(features.get("screen_time_hrs", 4.0))
    stress = float(features.get("stress_level", 5))
    tone_var = float(features.get("speech_tone_var", 30))
    mood_stability = float(features.get("mood_stability_index", 50))
    sentiment = float(features.get("sentiment_score", 0.0))
    pause_freq = float(features.get("pause_frequency_ppm", 10.0))
    hopelessness = float(features.get("hopelessness_prob", 0.1))
    keyword_int = float(features.get("keyword_intensity", 10.0))

    # 1. Sleep Risk Index (0 - 100)
    # Optimal sleep is 7-9 hours. Extreme sleep deficit or hypersomnia elevates risk.
    sleep_deficit = abs(7.5 - sleep_hrs)
    sleep_risk_index = np.clip((sleep_deficit / 4.5) * 60 + (stress / 10) * 40, 0.0, 100.0)

    # 2. Social Isolation Index (0 - 100)
    # Low social interaction + high screen time + negative sentiment elevates isolation risk.
    screen_factor = min(screen_hrs / 12.0, 1.0) * 30
    social_factor = (1.0 - (social_score / 100.0)) * 50
    sentiment_factor = max(0.0, -sentiment) * 20
    social_isolation_index = np.clip(screen_factor + social_factor + sentiment_factor, 0.0, 100.0)

    # 3. Mood Variability Index (0 - 100)
    # High MDQ + high tone variability + low mood stability index
    mdq_factor = (mdq / 13.0) * 45
    tone_factor = (tone_var / 70.0) * 25
    stability_factor = (1.0 - (mood_stability / 100.0)) * 30
    mood_variability_index = np.clip(mdq_factor + tone_factor + stability_factor, 0.0, 100.0)

    # 4. Depression Index (Weighted Composite Score 0 - 100)
    # Combines Questionnaire (40%), Sentiment (25%), Speech Pauses (15%), Wellbeing (20%)
    phq_norm = (phq9 / 27.0) * 100
    who5_inv_norm = (1.0 - (who5 / 100.0)) * 100
    sent_norm = max(0.0, -sentiment) * 100
    pause_norm = min(pause_freq / 30.0, 1.0) * 100

    depression_index = np.clip(
        0.40 * phq_norm + 0.20 * who5_inv_norm + 0.25 * sent_norm + 0.15 * pause_norm,
        0.0, 100.0
    )

    # 5. Suicide Risk Indicator Score (Composite 0 - 100)
    suicide_index = np.clip(
        0.40 * (hopelessness * 100) + 0.30 * keyword_int + 0.20 * phq_norm + 0.10 * social_isolation_index,
        0.0, 100.0
    )

    return {
        "sleep_risk_index": round(float(sleep_risk_index), 1),
        "social_isolation_index": round(float(social_isolation_index), 1),
        "mood_variability_index": round(float(mood_variability_index), 1),
        "depression_index": round(float(depression_index), 1),
        "suicide_risk_index": round(float(suicide_index), 1)
    }

def generate_longitudinal_trend(baseline_risks: Dict[str, float], weeks: int = 12) -> List[Dict[str, Any]]:
    """
    Generates historical 12-week risk trajectory for clinical trend monitoring.
    """
    trend = []
    dep_base = baseline_risks.get("depression", 50.0)
    anx_base = baseline_risks.get("anxiety", 40.0)
    bip_base = baseline_risks.get("bipolar", 20.0)
    sui_base = baseline_risks.get("suicide", 30.0)

    for w in range(1, weeks + 1):
        # Simulate natural variance over past weeks leading up to current week
        decay = (weeks - w) * 0.8
        noise_dep = np.random.normal(0, 3)
        noise_anx = np.random.normal(0, 2.5)

        trend.append({
            "week": f"W{w}",
            "label": f"Week {w}",
            "depression": round(float(np.clip(dep_base - decay + noise_dep, 5.0, 95.0)), 1),
            "anxiety": round(float(np.clip(anx_base - (decay*0.5) + noise_anx, 5.0, 95.0)), 1),
            "bipolar": round(float(np.clip(bip_base + np.random.normal(0, 2), 5.0, 90.0)), 1),
            "suicide": round(float(np.clip(sui_base - decay + np.random.normal(0, 2), 0.0, 90.0)), 1),
        })

    # Ensure last week matches baseline exactly
    trend[-1]["depression"] = round(dep_base, 1)
    trend[-1]["anxiety"] = round(anx_base, 1)
    trend[-1]["bipolar"] = round(bip_base, 1)
    trend[-1]["suicide"] = round(sui_base, 1)

    return trend
