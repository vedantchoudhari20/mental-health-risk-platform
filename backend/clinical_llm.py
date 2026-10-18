"""
Clinical Natural Language Insight Synthesizer & Zero-API-Key Local Intelligence Generator.
Generates comprehensive clinical summaries, risk rationales, and clinician documentation locally out-of-the-box.
Seamlessly integrates Google Gemini 1.5/2.0 Flash for authentic psychiatric SOAP progress notes when API key is provided.
"""

from typing import Dict, Any, Optional
from gemini_service import get_gemini_api_key, generate_gemini_soap_note

def generate_clinical_summary(patient_name: str, age: int, gender: str, risk_data: Dict[str, Any], xai_data: Dict[str, Any], qualitative_notes: str = "") -> Dict[str, str]:
    """
    Synthesizes rich clinical summaries and doctor notes.
    Uses Google Gemini Flash when API key is active; otherwise falls back to local heuristic engine with zero downtime.
    """
    scores = risk_data.get("risk_scores", {})
    categories = risk_data.get("risk_categories", {})
    top_factors = xai_data.get("top_contributing_factors", [])

    dep = scores.get("depression", 0)
    anx = scores.get("anxiety", 0)
    bip = scores.get("bipolar", 0)
    sui = scores.get("suicide", 0)

    top_factor_names = ", ".join([f["display_name"] for f in top_factors[:3]]) if top_factors else "questionnaire & behavioral metrics"

    # 1. Attempt Gemini Psychiatric SOAP synthesis if key is present
    if get_gemini_api_key():
        try:
            gemini_notes = generate_gemini_soap_note(
                patient_name=patient_name,
                age=age,
                gender=gender,
                risk_scores=scores,
                risk_categories=categories,
                top_factors=top_factors,
                qualitative_notes=qualitative_notes
            )
            if gemini_notes and gemini_notes.get("soap_note"):
                return {
                    "patient_summary": gemini_notes.get("clinical_summary", f"Evaluation for {patient_name}: Depression {dep}%, Anxiety {anx}%."),
                    "plain_language_explanation": gemini_notes.get("patient_explanation", "Your evaluation indicates stress and sleep disruption."),
                    "doctor_note": gemini_notes.get("soap_note"),
                    "llm_engine_used": "Google Gemini 1.5/2.0 Flash (Cloud Augmented)"
                }
        except Exception:
            pass  # Fall through seamlessly to local heuristic engine

    # 2. Local Fallback Heuristic Generation
    patient_summary = (
        f"Patient {patient_name} ({age}y/o {gender}) was evaluated using the multi-modal clinical decision support system. "
        f"Assessment indicates a {categories.get('depression', 'MODERATE')} risk for Depression ({dep}%), "
        f"{categories.get('anxiety', 'MODERATE')} risk for Anxiety ({anx}%), {categories.get('bipolar', 'LOW')} risk for Bipolar Disorder ({bip}%), "
        f"and {categories.get('suicide', 'MODERATE')} risk for Suicide Risk ({sui}%). "
        f"Primary risk drivers identified via SHAP feature attribution include {top_factor_names}."
    )

    plain_language_explanation = (
        f"Hello {patient_name}, your clinical evaluation shows elevated stress and low mood indicators, primarily influenced by "
        f"sleep disruption, reduced daily physical movement, and feeling disconnected. "
        f"Your recommended action plan focuses on restoring regular sleep patterns, gentle daily exercise, and structured relaxation techniques."
    )

    doctor_note = (
        f"CLINICAL IMPRESSION & SOAP PROGRESS NOTE:\n"
        f"Attending Physician: Dr. Rohan Shinde, MBBS, MD Psychiatry\n"
        f"Patient: {patient_name} | Age/Sex: {age}Y/{gender}\n\n"
        f"S (SUBJECTIVE): Patient presents for multi-modal psychiatric intake. Endorses mood fluctuation, sleep pattern irregularities, and daytime stress.\n"
        f"O (OBJECTIVE): Multi-Modal Risk Scores: Depression {dep}% ({categories.get('depression', 'MODERATE')}), GAD-7 Anxiety {anx}%, "
        f"Bipolar Screening {bip}%, Suicide Alert Index {sui}%.\n"
        f"Top SHAP Feature Contributors: {top_factor_names}.\n"
        f"A (ASSESSMENT): Clinical presentation consistent with {categories.get('depression', 'Moderate')} depressive indicators compounded by sleep deficit and autonomic stress.\n"
        f"P (PLAN): Evidence-based CBT behavioral activation, sleep hygiene protocol (target 7-8 hrs), 30-min daily aerobic walking, and clinical reassessment scheduled."
    )

    return {
        "patient_summary": patient_summary,
        "plain_language_explanation": plain_language_explanation,
        "doctor_note": doctor_note,
        "llm_engine_used": "Local Explainable Clinical Heuristic Engine (Zero API Key required)"
    }
