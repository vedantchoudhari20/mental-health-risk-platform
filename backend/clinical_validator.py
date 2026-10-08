"""
Clinical Validator Module — Input Validation, Consistency Checking & Data Quality Assessment.

Detects contradictions between self-reported data, flags insufficient evidence,
and computes an overall assessment quality grade before ML inference.
"""

import numpy as np
from typing import Dict, Any, List

from data_generator import FEATURE_COLUMNS


# ─────────────────────────────────────────────────────────
# Feature Domain Groups (for completeness assessment)
# ─────────────────────────────────────────────────────────

DOMAIN_GROUPS = {
    "questionnaire": {
        "features": ["phq9_score", "gad7_score", "mdq_score", "mood_stability_index", "who5_wellbeing"],
        "weight": 0.30,
        "label": "Standardized Questionnaire Scores",
    },
    "behavioral": {
        "features": ["sleep_hours", "social_interaction_score", "physical_activity_min",
                      "screen_time_hrs", "diet_quality_score", "stress_level"],
        "weight": 0.25,
        "label": "Behavioral & Lifestyle Indicators",
    },
    "speech_audio": {
        "features": ["speech_pitch_hz", "speech_tone_var", "speaking_rate_wpm",
                      "pause_frequency_ppm", "audio_energy"],
        "weight": 0.20,
        "label": "Speech & Audio Acoustic Features",
    },
    "text_nlp": {
        "features": ["sentiment_score", "sadness_prob", "anxiety_prob",
                      "hopelessness_prob", "keyword_intensity"],
        "weight": 0.15,
        "label": "Text NLP & Sentiment Analysis",
    },
    "demographics": {
        "features": ["age", "gender_code", "family_history", "past_episodes",
                      "medication_count", "medical_conditions"],
        "weight": 0.10,
        "label": "Demographics & Clinical History",
    },
}


# ─────────────────────────────────────────────────────────
# Contradiction Detection Rules
# ─────────────────────────────────────────────────────────

def _detect_contradictions(features: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Cross-checks feature values for clinically implausible combinations.
    Returns a list of detected contradictions with severity and explanation.
    """
    contradictions = []

    phq9 = float(features.get("phq9_score", -1))
    gad7 = float(features.get("gad7_score", -1))
    sleep = float(features.get("sleep_hours", -1))
    social = float(features.get("social_interaction_score", -1))
    activity = float(features.get("physical_activity_min", -1))
    stress = float(features.get("stress_level", -1))
    sentiment = float(features.get("sentiment_score", 0))
    who5 = float(features.get("who5_wellbeing", -1))
    hopelessness = float(features.get("hopelessness_prob", 0))

    # Rule 1: High PHQ-9 with healthy lifestyle indicators
    if phq9 >= 20 and sleep >= 7 and sleep <= 9 and activity >= 30 and social >= 60:
        contradictions.append({
            "id": "CONTRA_001",
            "severity": "WARNING",
            "rule": "Severe depression score with healthy behavioral profile",
            "detail": (
                f"PHQ-9 score ({phq9}) indicates severe depression, but sleep ({sleep} hrs), "
                f"activity ({activity} min), and social engagement ({social}/100) are all within healthy ranges. "
                "This combination is clinically unusual. Consider re-administering PHQ-9 or clinical interview."
            ),
            "affected_axes": ["depression"],
        })

    # Rule 2: Low PHQ-9 with severely disrupted lifestyle
    if 0 <= phq9 <= 4 and (sleep < 4 or activity < 5) and social < 15:
        contradictions.append({
            "id": "CONTRA_002",
            "severity": "WARNING",
            "rule": "Minimal depression score with severely disrupted lifestyle",
            "detail": (
                f"PHQ-9 score ({phq9}) suggests minimal depression, but sleep ({sleep} hrs), "
                f"activity ({activity} min), and social score ({social}/100) indicate significant impairment. "
                "Patient may be underreporting symptoms."
            ),
            "affected_axes": ["depression"],
        })

    # Rule 3: Extreme stress with positive sentiment
    if stress >= 9 and sentiment > 0.3:
        contradictions.append({
            "id": "CONTRA_003",
            "severity": "INFO",
            "rule": "Very high self-reported stress but positive text sentiment",
            "detail": (
                f"Stress level ({stress}/10) is very high, but text sentiment ({sentiment:.2f}) is positive. "
                "Possible: stress is situational (acute), patient is coping well verbally, "
                "or stress and text were reported at different time points."
            ),
            "affected_axes": ["anxiety"],
        })

    # Rule 4: WHO-5 contradiction with PHQ-9
    if who5 >= 0 and phq9 >= 0:
        if who5 >= 70 and phq9 >= 15:
            contradictions.append({
                "id": "CONTRA_004",
                "severity": "WARNING",
                "rule": "High wellbeing index contradicts high depression score",
                "detail": (
                    f"WHO-5 wellbeing ({who5}%) suggests good subjective wellbeing, "
                    f"but PHQ-9 ({phq9}) indicates moderately severe to severe depression. "
                    "These scales typically correlate inversely (r=-0.60, Topp et al. 2015)."
                ),
                "affected_axes": ["depression"],
            })
        elif who5 <= 28 and phq9 <= 4:
            contradictions.append({
                "id": "CONTRA_005",
                "severity": "INFO",
                "rule": "Poor wellbeing index contradicts minimal depression score",
                "detail": (
                    f"WHO-5 wellbeing ({who5}%) suggests poor wellbeing, "
                    f"but PHQ-9 ({phq9}) indicates minimal depression. "
                    "May reflect non-depressive factors affecting wellbeing (physical illness, life circumstances)."
                ),
                "affected_axes": ["depression"],
            })

    # Rule 5: Hopelessness without elevated depression
    if hopelessness > 0.6 and phq9 >= 0 and phq9 < 10:
        contradictions.append({
            "id": "CONTRA_006",
            "severity": "CRITICAL",
            "rule": "High hopelessness detected without matching depression severity",
            "detail": (
                f"Hopelessness probability ({hopelessness:.2f}) is elevated, which is a key suicide risk factor, "
                f"but PHQ-9 ({phq9}) does not indicate clinical depression. "
                "Hopelessness can independently predict suicide risk even without current MDD. "
                "Immediate safety assessment recommended."
            ),
            "affected_axes": ["suicide"],
        })

    # Rule 6: Sleep extreme values
    if sleep > 0 and (sleep < 3 or sleep > 12):
        contradictions.append({
            "id": "CONTRA_007",
            "severity": "INFO",
            "rule": "Extreme sleep duration reported",
            "detail": (
                f"Reported sleep of {sleep} hrs is outside the typical range (4-10 hrs). "
                "Verify this is not a data entry error."
            ),
            "affected_axes": ["depression", "anxiety"],
        })

    return contradictions


# ─────────────────────────────────────────────────────────
# Input Completeness Assessment
# ─────────────────────────────────────────────────────────

def _assess_completeness(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates how many required features are provided and which domains are covered.
    """
    total_features = len(FEATURE_COLUMNS)
    provided = 0
    missing = []
    default_suspected = []

    for col in FEATURE_COLUMNS:
        val = features.get(col)
        if val is not None and val != 0.0:
            provided += 1
        elif val == 0.0 and col not in ("family_history", "medical_conditions", "medication_count", "past_episodes"):
            # 0.0 for most features is likely a default/missing value
            default_suspected.append(col)
            provided += 1  # Count as provided but flag
        elif val is None:
            missing.append(col)
        else:
            provided += 1

    domain_coverage = {}
    for domain_name, domain_info in DOMAIN_GROUPS.items():
        domain_features = domain_info["features"]
        domain_provided = sum(1 for f in domain_features if features.get(f) is not None)
        domain_total = len(domain_features)
        coverage_pct = round((domain_provided / domain_total) * 100, 1) if domain_total > 0 else 0
        domain_coverage[domain_name] = {
            "label": domain_info["label"],
            "provided": domain_provided,
            "total": domain_total,
            "coverage_pct": coverage_pct,
            "weight": domain_info["weight"],
            "missing": [f for f in domain_features if features.get(f) is None],
        }

    overall_pct = round((provided / total_features) * 100, 1) if total_features > 0 else 0

    # Weighted completeness (some domains matter more)
    weighted_score = sum(
        dc["coverage_pct"] / 100 * dc["weight"]
        for dc in domain_coverage.values()
    )
    weighted_pct = round(weighted_score / sum(d["weight"] for d in DOMAIN_GROUPS.values()) * 100, 1)

    return {
        "total_features": total_features,
        "features_provided": provided,
        "features_missing": len(missing),
        "missing_features": missing,
        "default_suspected": default_suspected,
        "overall_completeness_pct": overall_pct,
        "weighted_completeness_pct": weighted_pct,
        "domain_coverage": domain_coverage,
    }


# ─────────────────────────────────────────────────────────
# Data Sufficiency per Risk Axis
# ─────────────────────────────────────────────────────────

AXIS_REQUIRED_FEATURES = {
    "depression": {
        "critical": ["phq9_score", "sleep_hours", "social_interaction_score", "sentiment_score"],
        "important": ["who5_wellbeing", "physical_activity_min", "stress_level", "sadness_prob"],
        "supplementary": ["speech_pitch_hz", "pause_frequency_ppm", "diet_quality_score"],
    },
    "anxiety": {
        "critical": ["gad7_score", "stress_level", "anxiety_prob"],
        "important": ["sleep_hours", "sentiment_score", "screen_time_hrs"],
        "supplementary": ["speech_tone_var", "speaking_rate_wpm", "social_interaction_score"],
    },
    "bipolar": {
        "critical": ["mdq_score", "mood_stability_index"],
        "important": ["sleep_hours", "speech_tone_var", "speaking_rate_wpm"],
        "supplementary": ["past_episodes", "family_history", "audio_energy"],
    },
    "suicide": {
        "critical": ["hopelessness_prob", "keyword_intensity", "sentiment_score"],
        "important": ["phq9_score", "social_interaction_score", "past_episodes"],
        "supplementary": ["sleep_hours", "family_history", "stress_level"],
    },
}


def _assess_axis_sufficiency(features: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    For each risk axis, evaluates whether there is sufficient data for a credible assessment.
    """
    axis_sufficiency = {}

    for axis, requirements in AXIS_REQUIRED_FEATURES.items():
        critical_provided = sum(1 for f in requirements["critical"] if features.get(f) is not None)
        critical_total = len(requirements["critical"])
        important_provided = sum(1 for f in requirements["important"] if features.get(f) is not None)
        important_total = len(requirements["important"])
        supp_provided = sum(1 for f in requirements["supplementary"] if features.get(f) is not None)
        supp_total = len(requirements["supplementary"])

        critical_pct = critical_provided / critical_total if critical_total > 0 else 0
        important_pct = important_provided / important_total if important_total > 0 else 0
        supp_pct = supp_provided / supp_total if supp_total > 0 else 0

        # Weighted sufficiency: critical 60%, important 30%, supplementary 10%
        sufficiency_score = round((critical_pct * 0.6 + important_pct * 0.3 + supp_pct * 0.1) * 100, 1)

        if critical_pct < 0.5:
            grade = "INSUFFICIENT"
        elif critical_pct < 1.0 or sufficiency_score < 50:
            grade = "PARTIAL"
        elif sufficiency_score < 75:
            grade = "ADEQUATE"
        else:
            grade = "STRONG"

        missing_critical = [f for f in requirements["critical"] if features.get(f) is None]

        axis_sufficiency[axis] = {
            "sufficiency_score": sufficiency_score,
            "grade": grade,
            "critical_coverage": f"{critical_provided}/{critical_total}",
            "important_coverage": f"{important_provided}/{important_total}",
            "supplementary_coverage": f"{supp_provided}/{supp_total}",
            "missing_critical_features": missing_critical,
            "recommendation": (
                f"Assessment for {axis} risk has {grade.lower()} evidence. "
                + (f"Missing critical features: {', '.join(missing_critical)}. " if missing_critical else "")
                + ("Clinical confirmation strongly recommended." if grade in ("INSUFFICIENT", "PARTIAL") else "")
            ),
        }

    return axis_sufficiency


# ─────────────────────────────────────────────────────────
# Overall Validation Report
# ─────────────────────────────────────────────────────────

def validate_patient_input(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs complete validation pipeline on patient feature input:
    1. Completeness assessment (what % of fields are provided)
    2. Contradiction detection (cross-domain consistency checks)
    3. Per-axis data sufficiency (is there enough evidence for each risk axis)
    4. Overall quality grade

    Returns a structured validation report that should accompany any prediction.
    """
    completeness = _assess_completeness(features)
    contradictions = _detect_contradictions(features)
    axis_sufficiency = _assess_axis_sufficiency(features)

    # Overall quality grade
    weighted_completeness = completeness["weighted_completeness_pct"]
    critical_contradictions = sum(1 for c in contradictions if c["severity"] == "CRITICAL")
    warning_contradictions = sum(1 for c in contradictions if c["severity"] == "WARNING")
    avg_sufficiency = np.mean([a["sufficiency_score"] for a in axis_sufficiency.values()])

    # Compute quality grade (A/B/C/D/F)
    quality_score = weighted_completeness * 0.4 + avg_sufficiency * 0.4
    quality_score -= critical_contradictions * 15
    quality_score -= warning_contradictions * 5

    if quality_score >= 85:
        quality_grade = "A"
        quality_label = "High confidence — comprehensive multi-modal data with consistent indicators"
    elif quality_score >= 70:
        quality_grade = "B"
        quality_label = "Moderate confidence — adequate data with minor gaps or inconsistencies"
    elif quality_score >= 50:
        quality_grade = "C"
        quality_label = "Limited confidence — significant data gaps; results should be interpreted cautiously"
    elif quality_score >= 30:
        quality_grade = "D"
        quality_label = "Low confidence — insufficient data for reliable assessment; clinical evaluation essential"
    else:
        quality_grade = "F"
        quality_label = "Insufficient — data quality too low for meaningful assessment; requires clinical intake"

    # Confidence adjustment factor (0.0 to 1.0) to multiply against model confidence
    confidence_adjustment = np.clip(quality_score / 100.0, 0.1, 1.0)

    return {
        "quality_grade": quality_grade,
        "quality_label": quality_label,
        "quality_score": round(float(quality_score), 1),
        "confidence_adjustment": round(float(confidence_adjustment), 3),
        "completeness": completeness,
        "contradictions": contradictions,
        "contradiction_count": len(contradictions),
        "critical_contradictions": critical_contradictions,
        "axis_sufficiency": axis_sufficiency,
        "overall_recommendation": (
            "All risk assessments meet data sufficiency thresholds."
            if all(a["grade"] in ("STRONG", "ADEQUATE") for a in axis_sufficiency.values())
            else "Some risk axes have insufficient evidence. See axis_sufficiency for details."
        ),
    }
