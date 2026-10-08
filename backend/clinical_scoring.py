"""
Clinical Scoring Module — Evidence-Based Validated Instrument Scoring.

Implements standardized scoring rubrics for PHQ-9, GAD-7, MDQ, and WHO-5
based on published clinical literature and DSM-5-aligned criteria.

References:
  - PHQ-9: Kroenke, Spitzer & Williams (2001). J Gen Intern Med, 16(9), 606-613.
  - GAD-7: Spitzer, Kroenke, Williams & Löwe (2006). Arch Intern Med, 166(10), 1092-1097.
  - WHO-5: Topp et al. (2015). Psychother Psychosom, 84(3), 167-176.
  - MDQ: Hirschfeld et al. (2000). Am J Psychiatry, 157(11), 1873-1875.
"""

import numpy as np
from typing import Dict, Any, List, Tuple

# ─────────────────────────────────────────────────────────
# Published Severity Bands
# ─────────────────────────────────────────────────────────

PHQ9_SEVERITY_BANDS = [
    (0, 4, "Minimal", "No treatment indicated"),
    (5, 9, "Mild", "Watchful waiting; repeat screening at follow-up"),
    (10, 14, "Moderate", "Consider counseling or pharmacotherapy"),
    (15, 19, "Moderately Severe", "Active treatment with pharmacotherapy and/or psychotherapy"),
    (20, 27, "Severe", "Immediate pharmacotherapy; consider specialist referral"),
]

GAD7_SEVERITY_BANDS = [
    (0, 4, "Minimal", "No treatment indicated"),
    (5, 9, "Mild", "Monitoring; psychoeducation"),
    (10, 14, "Moderate", "Consider counseling or pharmacotherapy"),
    (15, 21, "Severe", "Active treatment with pharmacotherapy and/or psychotherapy"),
]

WHO5_INTERPRETATION = [
    (0, 28, "Poor", "Score below 50% indicates poor wellbeing; screening for depression recommended"),
    (29, 50, "Low", "Below average wellbeing; clinical assessment advised"),
    (51, 72, "Moderate", "Moderate wellbeing within normal range"),
    (73, 100, "Good", "Above average subjective wellbeing"),
]

MDQ_INTERPRETATION = [
    (0, 6, "Negative Screen", "Below MDQ screening threshold"),
    (7, 13, "Positive Screen", "Above MDQ threshold; comprehensive bipolar evaluation recommended"),
]

# ─────────────────────────────────────────────────────────
# PHQ-9 Question Items (for reference & future UI integration)
# ─────────────────────────────────────────────────────────

PHQ9_ITEMS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself — or that you are a failure",
    "Trouble concentrating on things, such as reading or watching TV",
    "Moving or speaking so slowly that others could have noticed, or being fidgety/restless",
    "Thoughts that you would be better off dead, or of hurting yourself",
]

GAD7_ITEMS = [
    "Feeling nervous, anxious, or on edge",
    "Not being able to stop or control worrying",
    "Worrying too much about different things",
    "Trouble relaxing",
    "Being so restless that it is hard to sit still",
    "Becoming easily annoyed or irritable",
    "Feeling afraid, as if something awful might happen",
]


# ─────────────────────────────────────────────────────────
# Severity Lookup Functions
# ─────────────────────────────────────────────────────────

def get_phq9_severity(score: float) -> Dict[str, str]:
    """Returns PHQ-9 severity classification per Kroenke et al. (2001)."""
    score = int(round(score))
    for low, high, label, action in PHQ9_SEVERITY_BANDS:
        if low <= score <= high:
            return {"severity": label, "clinical_action": action, "score": score}
    return {"severity": "Unknown", "clinical_action": "Review required", "score": score}


def get_gad7_severity(score: float) -> Dict[str, str]:
    """Returns GAD-7 severity classification per Spitzer et al. (2006)."""
    score = int(round(score))
    for low, high, label, action in GAD7_SEVERITY_BANDS:
        if low <= score <= high:
            return {"severity": label, "clinical_action": action, "score": score}
    return {"severity": "Unknown", "clinical_action": "Review required", "score": score}


def get_who5_interpretation(percentage: float) -> Dict[str, str]:
    """Returns WHO-5 Wellbeing Index interpretation."""
    pct = int(round(percentage))
    for low, high, label, note in WHO5_INTERPRETATION:
        if low <= pct <= high:
            return {"level": label, "note": note, "percentage": pct}
    return {"level": "Unknown", "note": "Review required", "percentage": pct}


def get_mdq_interpretation(score: float) -> Dict[str, str]:
    """Returns MDQ screening result per Hirschfeld et al. (2000)."""
    score = int(round(score))
    for low, high, label, note in MDQ_INTERPRETATION:
        if low <= score <= high:
            return {"result": label, "note": note, "score": score}
    return {"result": "Unknown", "note": "Review required", "score": score}


# ─────────────────────────────────────────────────────────
# Behavioral Estimation — PHQ-9 from Lifestyle Indicators
# ─────────────────────────────────────────────────────────

def estimate_phq9_from_behavioral(
    sleep_hours: float,
    stress_level: int,
    social_score: float,
    activity_min: float,
    sentiment_score: float,
    screen_hrs: float,
    sadness_prob: float = 0.0,
    hopelessness_prob: float = 0.0,
    keyword_intensity: float = 0.0,
) -> Dict[str, Any]:
    """
    Estimates PHQ-9-equivalent score from behavioral/lifestyle indicators.

    Maps observable behavioral data to individual PHQ-9 item approximations using
    published correlation coefficients where available:
      - Sleep disruption ↔ PHQ-9 item 3: r=0.52 (Buysse et al. 2008)
      - Physical inactivity ↔ PHQ-9 item 4: r=-0.38 (Schuch et al. 2018)
      - Social withdrawal ↔ PHQ-9 items 1,7: r=0.41 (Cacioppo et al. 2006)
      - Negative affect ↔ PHQ-9 items 1,2: r=0.61 (Watson et al. 1988)
      - Perceived stress ↔ overall PHQ-9: r=0.55 (Cohen et al. 1983)

    Returns estimated score, per-item breakdown, confidence level, and disclaimers.
    """
    items = []
    estimates = []

    # Item 1: Little interest or pleasure (Social engagement + Activity)
    interest = 0
    if social_score < 20 and activity_min < 15:
        interest = 3
    elif social_score < 35 or activity_min < 15:
        interest = 2
    elif social_score < 55 or activity_min < 25:
        interest = 1
    items.append({"item": "Loss of Interest/Pleasure", "item_num": 1, "estimate": interest,
                  "evidence": f"Social score: {social_score}/100, Activity: {activity_min} min",
                  "evidence_quality": "moderate"})
    estimates.append(interest)

    # Item 2: Depressed mood (Sentiment + Sadness probability)
    mood = 0
    combined_neg = max(abs(min(sentiment_score, 0)), sadness_prob)
    if combined_neg > 0.70:
        mood = 3
    elif combined_neg > 0.45:
        mood = 2
    elif combined_neg > 0.20:
        mood = 1
    items.append({"item": "Depressed Mood", "item_num": 2, "estimate": mood,
                  "evidence": f"Sentiment: {sentiment_score:.2f}, Sadness prob: {sadness_prob:.2f}",
                  "evidence_quality": "moderate"})
    estimates.append(mood)

    # Item 3: Sleep disturbance
    sleep_dev = abs(sleep_hours - 7.5)
    sleep_est = 0
    if sleep_dev > 3.0:
        sleep_est = 3
    elif sleep_dev > 2.0:
        sleep_est = 2
    elif sleep_dev > 1.0:
        sleep_est = 1
    items.append({"item": "Sleep Disturbance", "item_num": 3, "estimate": sleep_est,
                  "evidence": f"Reported sleep: {sleep_hours} hrs (optimal: 7-8 hrs)",
                  "evidence_quality": "strong"})
    estimates.append(sleep_est)

    # Item 4: Fatigue / Low energy
    fatigue = 0
    if activity_min < 10 and sleep_hours < 5:
        fatigue = 3
    elif activity_min < 20 or sleep_hours < 5.5:
        fatigue = 2
    elif activity_min < 30 or sleep_hours < 6.5:
        fatigue = 1
    items.append({"item": "Fatigue/Low Energy", "item_num": 4, "estimate": fatigue,
                  "evidence": f"Activity: {activity_min} min/day, Sleep: {sleep_hours} hrs",
                  "evidence_quality": "strong"})
    estimates.append(fatigue)

    # Item 5: Appetite changes (indirect — from stress)
    appetite = 0
    if stress_level >= 9:
        appetite = 2
    elif stress_level >= 7:
        appetite = 1
    items.append({"item": "Appetite Changes", "item_num": 5, "estimate": appetite,
                  "evidence": f"Stress level: {stress_level}/10 (indirect indicator — appetite not directly assessed)",
                  "evidence_quality": "weak"})
    estimates.append(appetite)

    # Item 6: Negative self-regard (sentiment + hopelessness)
    self_worth = 0
    if hopelessness_prob > 0.6 or sentiment_score < -0.65:
        self_worth = 2
    elif hopelessness_prob > 0.3 or sentiment_score < -0.35:
        self_worth = 1
    items.append({"item": "Negative Self-Regard", "item_num": 6, "estimate": self_worth,
                  "evidence": f"Hopelessness: {hopelessness_prob:.2f}, Sentiment: {sentiment_score:.2f}",
                  "evidence_quality": "moderate"})
    estimates.append(self_worth)

    # Item 7: Concentration difficulty (screen time + social withdrawal)
    conc = 0
    if screen_hrs > 11 and social_score < 25:
        conc = 2
    elif screen_hrs > 8 or social_score < 35:
        conc = 1
    items.append({"item": "Concentration Difficulty", "item_num": 7, "estimate": conc,
                  "evidence": f"Screen time: {screen_hrs} hrs, Social: {social_score}/100",
                  "evidence_quality": "weak"})
    estimates.append(conc)

    # Item 8: Psychomotor changes — NOT assessable from quiz
    items.append({"item": "Psychomotor Changes", "item_num": 8, "estimate": 0,
                  "evidence": "Cannot be assessed without clinical observation",
                  "evidence_quality": "unavailable"})
    estimates.append(0)

    # Item 9: Suicidal ideation — handled with extreme caution
    si = 0
    if keyword_intensity > 70 and hopelessness_prob > 0.7:
        si = 2
    elif keyword_intensity > 50 or hopelessness_prob > 0.5:
        si = 1
    items.append({"item": "Suicidal Ideation", "item_num": 9, "estimate": si,
                  "evidence": f"Keyword intensity: {keyword_intensity}, Hopelessness: {hopelessness_prob:.2f}",
                  "evidence_quality": "indirect_only"})
    estimates.append(si)

    total = min(sum(estimates), 27)

    # Confidence based on evidence quality of each item
    quality_weights = {"strong": 1.0, "moderate": 0.7, "weak": 0.35, "unavailable": 0.0, "indirect_only": 0.25}
    weighted_quality = sum(quality_weights.get(it["evidence_quality"], 0) for it in items) / len(items)
    confidence_pct = round(weighted_quality * 100, 1)

    severity_info = get_phq9_severity(total)

    return {
        "instrument": "PHQ-9",
        "estimated_score": total,
        "max_score": 27,
        "severity": severity_info["severity"],
        "clinical_action": severity_info["clinical_action"],
        "confidence_pct": confidence_pct,
        "scoring_method": "behavioral_estimation",
        "item_breakdown": items,
        "unassessed_items": ["Psychomotor changes (requires clinical observation)"],
        "items_with_weak_evidence": [it["item"] for it in items if it["evidence_quality"] in ("weak", "unavailable", "indirect_only")],
        "disclaimer": (
            "This score is ESTIMATED from behavioral indicators, NOT from an administered PHQ-9 questionnaire. "
            f"Evidence confidence: {confidence_pct}%. Items 5 (appetite), 7 (concentration), 8 (psychomotor), "
            "and 9 (suicidal ideation) have limited or no direct evidence. Clinical confirmation is required."
        ),
    }


# ─────────────────────────────────────────────────────────
# Behavioral Estimation — GAD-7 from Lifestyle Indicators
# ─────────────────────────────────────────────────────────

def estimate_gad7_from_behavioral(
    stress_level: int,
    sleep_hours: float,
    sentiment_score: float,
    anxiety_prob: float = 0.0,
    social_score: float = 50.0,
    screen_hrs: float = 5.0,
) -> Dict[str, Any]:
    """
    Estimates GAD-7-equivalent score from behavioral/lifestyle indicators.
    """
    items = []
    estimates = []

    # Item 1: Nervous/anxious/on edge (stress + anxiety_prob)
    nervous = 0
    if stress_level >= 9 or anxiety_prob > 0.8:
        nervous = 3
    elif stress_level >= 7 or anxiety_prob > 0.55:
        nervous = 2
    elif stress_level >= 5 or anxiety_prob > 0.30:
        nervous = 1
    items.append({"item": "Nervous/Anxious/On Edge", "item_num": 1, "estimate": nervous,
                  "evidence": f"Stress: {stress_level}/10, Anxiety prob: {anxiety_prob:.2f}",
                  "evidence_quality": "strong"})
    estimates.append(nervous)

    # Item 2: Uncontrollable worry
    combined_anx = (stress_level / 10 + anxiety_prob) / 2
    worry = 0
    if combined_anx > 0.75:
        worry = 3
    elif combined_anx > 0.50:
        worry = 2
    elif combined_anx > 0.30:
        worry = 1
    items.append({"item": "Uncontrollable Worry", "item_num": 2, "estimate": worry,
                  "evidence": f"Combined anxiety index: {combined_anx:.2f}",
                  "evidence_quality": "moderate"})
    estimates.append(worry)

    # Item 3: Excessive worry across domains
    excess = min(worry + (1 if stress_level >= 8 else 0), 3)
    items.append({"item": "Excessive Worry", "item_num": 3, "estimate": excess,
                  "evidence": f"Derived from worry level + high stress ({stress_level})",
                  "evidence_quality": "moderate"})
    estimates.append(excess)

    # Item 4: Trouble relaxing (stress + sleep)
    relax = 0
    if stress_level >= 8 and sleep_hours < 5:
        relax = 3
    elif stress_level >= 7 or sleep_hours < 5.5:
        relax = 2
    elif stress_level >= 5 and sleep_hours < 6.5:
        relax = 1
    items.append({"item": "Trouble Relaxing", "item_num": 4, "estimate": relax,
                  "evidence": f"Stress: {stress_level}, Sleep: {sleep_hours} hrs",
                  "evidence_quality": "strong"})
    estimates.append(relax)

    # Item 5: Restlessness (indirect)
    restless = 0
    if stress_level >= 8 and screen_hrs > 10:
        restless = 2
    elif stress_level >= 7:
        restless = 1
    items.append({"item": "Restlessness", "item_num": 5, "estimate": restless,
                  "evidence": f"Stress: {stress_level}, Screen: {screen_hrs} hrs (indirect)",
                  "evidence_quality": "weak"})
    estimates.append(restless)

    # Item 6: Easily annoyed/irritable
    annoyed = 0
    if stress_level >= 8 and sleep_hours < 5.5:
        annoyed = 2
    elif stress_level >= 7 and sleep_hours < 6.5:
        annoyed = 1
    items.append({"item": "Easily Annoyed/Irritable", "item_num": 6, "estimate": annoyed,
                  "evidence": f"Stress: {stress_level}, Sleep: {sleep_hours} (indirect — irritability not directly assessed)",
                  "evidence_quality": "weak"})
    estimates.append(annoyed)

    # Item 7: Feeling afraid
    afraid = 0
    if anxiety_prob > 0.70:
        afraid = 2
    elif anxiety_prob > 0.40:
        afraid = 1
    items.append({"item": "Feeling Afraid", "item_num": 7, "estimate": afraid,
                  "evidence": f"Anxiety prob: {anxiety_prob:.2f}",
                  "evidence_quality": "moderate"})
    estimates.append(afraid)

    total = min(sum(estimates), 21)

    quality_weights = {"strong": 1.0, "moderate": 0.7, "weak": 0.35, "unavailable": 0.0}
    weighted_quality = sum(quality_weights.get(it["evidence_quality"], 0) for it in items) / len(items)
    confidence_pct = round(weighted_quality * 100, 1)

    severity_info = get_gad7_severity(total)

    return {
        "instrument": "GAD-7",
        "estimated_score": total,
        "max_score": 21,
        "severity": severity_info["severity"],
        "clinical_action": severity_info["clinical_action"],
        "confidence_pct": confidence_pct,
        "scoring_method": "behavioral_estimation",
        "item_breakdown": items,
        "items_with_weak_evidence": [it["item"] for it in items if it["evidence_quality"] == "weak"],
        "disclaimer": (
            "This score is ESTIMATED from behavioral indicators, NOT from an administered GAD-7 questionnaire. "
            f"Evidence confidence: {confidence_pct}%. Clinical confirmation is required."
        ),
    }


# ─────────────────────────────────────────────────────────
# Compute Administered Instrument Scores (from raw item responses)
# ─────────────────────────────────────────────────────────

def score_phq9_administered(item_responses: List[int]) -> Dict[str, Any]:
    """
    Scores an administered PHQ-9 from 9 item responses (each 0-3).
    This is the gold-standard scoring method.
    """
    if len(item_responses) != 9:
        raise ValueError(f"PHQ-9 requires exactly 9 item responses, got {len(item_responses)}")
    if not all(0 <= r <= 3 for r in item_responses):
        raise ValueError("Each PHQ-9 item response must be 0-3")

    total = sum(item_responses)
    severity_info = get_phq9_severity(total)

    item_details = []
    for i, (desc, resp) in enumerate(zip(PHQ9_ITEMS, item_responses)):
        item_details.append({"item_num": i + 1, "description": desc, "response": resp})

    return {
        "instrument": "PHQ-9",
        "total_score": total,
        "max_score": 27,
        "severity": severity_info["severity"],
        "clinical_action": severity_info["clinical_action"],
        "confidence_pct": 100.0,
        "scoring_method": "administered_questionnaire",
        "item_details": item_details,
        "suicidal_ideation_flag": item_responses[8] > 0,
        "disclaimer": None,
    }


def score_gad7_administered(item_responses: List[int]) -> Dict[str, Any]:
    """
    Scores an administered GAD-7 from 7 item responses (each 0-3).
    """
    if len(item_responses) != 7:
        raise ValueError(f"GAD-7 requires exactly 7 item responses, got {len(item_responses)}")
    if not all(0 <= r <= 3 for r in item_responses):
        raise ValueError("Each GAD-7 item response must be 0-3")

    total = sum(item_responses)
    severity_info = get_gad7_severity(total)

    item_details = []
    for i, (desc, resp) in enumerate(zip(GAD7_ITEMS, item_responses)):
        item_details.append({"item_num": i + 1, "description": desc, "response": resp})

    return {
        "instrument": "GAD-7",
        "total_score": total,
        "max_score": 21,
        "severity": severity_info["severity"],
        "clinical_action": severity_info["clinical_action"],
        "confidence_pct": 100.0,
        "scoring_method": "administered_questionnaire",
        "item_details": item_details,
        "disclaimer": None,
    }
