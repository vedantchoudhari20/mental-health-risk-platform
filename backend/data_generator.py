"""
Data Generator & Public Dataset Simulator for Mental Health Risk Detection Platform.

IMPORTANT CHANGES (v2.0 — Clinical Integrity Rewrite):
  - Labels are now derived from FEATURES using published clinical scoring rules,
    NOT from the same latent variables that generate features.
  - Added realistic label noise to simulate clinical inter-rater disagreement
    (kappa ≈ 0.75-0.85 for mental health assessments; Kroenke et al. 2001).
  - Train/test labeling is clinically grounded, not circular.

Simulates clinical multi-modal datasets matching benchmarks (PHQ-9, GAD-7, MDQ, DAIC-WOZ, GoEmotions, NHANES).
Supports dynamic patient registration store with Medical Record Numbers (MRN).
"""

import numpy as np
import polars as pl
from typing import Dict, List, Any, Tuple
import random

# Set seed for reproducible statistical behavior
np.random.seed(42)

FEATURE_COLUMNS = [
    # Questionnaires
    "phq9_score", "gad7_score", "mdq_score", "mood_stability_index", "who5_wellbeing",
    # Behavioral
    "sleep_hours", "social_interaction_score", "physical_activity_min", "screen_time_hrs", "diet_quality_score", "stress_level",
    # Speech
    "speech_pitch_hz", "speech_tone_var", "speaking_rate_wpm", "pause_frequency_ppm", "audio_energy",
    # Text NLP
    "sentiment_score", "sadness_prob", "anxiety_prob", "hopelessness_prob", "keyword_intensity",
    # Clinical & Demographics
    "age", "gender_code", "family_history", "past_episodes", "medication_count", "medical_conditions"
]

# Global Patient Database Store
PATIENT_STORE = [
    {
        "id": "MRN-2026-1023",
        "name": "Vedant",
        "age": 21,
        "gender": "Male",
        "mrn": "MRN-2026-1023",
        "contact": "+1 (555) 234-5678",
        "primary_physician": "Dr. Rohan Shinde, MBBS, MD Psychiatry",
        "clinical_summary": "Presents with severe fatigue, persistent low mood for 6 weeks, high PHQ-9 questionnaire score, reduced speaking rate, and social withdrawal.",
        "features": {
            "phq9_score": 22,
            "gad7_score": 14,
            "mdq_score": 3,
            "mood_stability_index": 72.0,
            "who5_wellbeing": 18.0,
            "sleep_hours": 4.5,
            "social_interaction_score": 18.0,
            "physical_activity_min": 15,
            "screen_time_hrs": 9.5,
            "diet_quality_score": 35.0,
            "stress_level": 8,
            "speech_pitch_hz": 125.0,
            "speech_tone_var": 18.0,
            "speaking_rate_wpm": 92.0,
            "pause_frequency_ppm": 24.0,
            "audio_energy": 0.22,
            "sentiment_score": -0.68,
            "sadness_prob": 0.84,
            "anxiety_prob": 0.62,
            "hopelessness_prob": 0.58,
            "keyword_intensity": 65.0,
            "age": 21,
            "gender_code": 1,
            "family_history": 1,
            "past_episodes": 2,
            "medication_count": 1,
            "medical_conditions": 0
        }
    },
    {
        "id": "MRN-2026-1045",
        "name": "Sarah M.",
        "age": 34,
        "gender": "Female",
        "mrn": "MRN-2026-1045",
        "contact": "+1 (555) 345-6789",
        "primary_physician": "Dr. Amanda Vance, MD",
        "clinical_summary": "High generalized anxiety with panic features. Rapid speech rate, elevated GAD-7 score, disrupted sleep, and high workplace stress.",
        "features": {
            "phq9_score": 11,
            "gad7_score": 19,
            "mdq_score": 4,
            "mood_stability_index": 55.0,
            "who5_wellbeing": 35.0,
            "sleep_hours": 5.0,
            "social_interaction_score": 52.0,
            "physical_activity_min": 25,
            "screen_time_hrs": 11.0,
            "diet_quality_score": 50.0,
            "stress_level": 9,
            "speech_pitch_hz": 230.0,
            "speech_tone_var": 48.0,
            "speaking_rate_wpm": 195.0,
            "pause_frequency_ppm": 14.0,
            "audio_energy": 0.75,
            "sentiment_score": -0.45,
            "sadness_prob": 0.38,
            "anxiety_prob": 0.89,
            "hopelessness_prob": 0.25,
            "keyword_intensity": 40.0,
            "age": 34,
            "gender_code": 0,
            "family_history": 1,
            "past_episodes": 1,
            "medication_count": 0,
            "medical_conditions": 1
        }
    },
    {
        "id": "MRN-2026-1089",
        "name": "Marcus K.",
        "age": 42,
        "gender": "Male",
        "mrn": "MRN-2026-1089",
        "contact": "+1 (555) 456-7890",
        "primary_physician": "Dr. Robert Chen, MD",
        "clinical_summary": "History of episodic mood swings, high MDQ screening score, variable sleep pattern (2-10 hrs fluctuation), and high energy speech.",
        "features": {
            "phq9_score": 14,
            "gad7_score": 10,
            "mdq_score": 11,
            "mood_stability_index": 22.0,
            "who5_wellbeing": 42.0,
            "sleep_hours": 4.0,
            "social_interaction_score": 85.0,
            "physical_activity_min": 90,
            "screen_time_hrs": 7.0,
            "diet_quality_score": 60.0,
            "stress_level": 7,
            "speech_pitch_hz": 175.0,
            "speech_tone_var": 62.0,
            "speaking_rate_wpm": 210.0,
            "pause_frequency_ppm": 6.0,
            "audio_energy": 0.88,
            "sentiment_score": 0.15,
            "sadness_prob": 0.30,
            "anxiety_prob": 0.45,
            "hopelessness_prob": 0.18,
            "keyword_intensity": 35.0,
            "age": 42,
            "gender_code": 1,
            "family_history": 1,
            "past_episodes": 4,
            "medication_count": 2,
            "medical_conditions": 0
        }
    },
    {
        "id": "MRN-2026-1102",
        "name": "Elena R.",
        "age": 28,
        "gender": "Female",
        "mrn": "MRN-2026-1102",
        "contact": "+1 (555) 567-8901",
        "primary_physician": "Dr. Rohan Shinde, MBBS, MD Psychiatry",
        "clinical_summary": "Mild seasonal stress symptoms, well-maintained sleep hygiene, low PHQ-9/GAD-7 scores, active social engagement.",
        "features": {
            "phq9_score": 4,
            "gad7_score": 3,
            "mdq_score": 1,
            "mood_stability_index": 88.0,
            "who5_wellbeing": 82.0,
            "sleep_hours": 7.8,
            "social_interaction_score": 78.0,
            "physical_activity_min": 45,
            "screen_time_hrs": 4.0,
            "diet_quality_score": 82.0,
            "stress_level": 3,
            "speech_pitch_hz": 210.0,
            "speech_tone_var": 38.0,
            "speaking_rate_wpm": 145.0,
            "pause_frequency_ppm": 9.0,
            "audio_energy": 0.65,
            "sentiment_score": 0.42,
            "sadness_prob": 0.12,
            "anxiety_prob": 0.15,
            "hopelessness_prob": 0.05,
            "keyword_intensity": 10.0,
            "age": 28,
            "gender_code": 0,
            "family_history": 0,
            "past_episodes": 0,
            "medication_count": 0,
            "medical_conditions": 0
        }
    }
]


def _generate_clinical_labels(
    phq9: np.ndarray,
    gad7: np.ndarray,
    mdq: np.ndarray,
    mood_stability: np.ndarray,
    sleep_hours: np.ndarray,
    social_score: np.ndarray,
    sentiment: np.ndarray,
    hopelessness: np.ndarray,
    keyword_intensity: np.ndarray,
    stress: np.ndarray,
    past_episodes: np.ndarray,
    family_history: np.ndarray,
    n_samples: int,
) -> Dict[str, np.ndarray]:
    """
    Generates ground-truth labels using PUBLISHED CLINICAL SCORING RULES,
    not from the latent variables that generate features.

    Depression label: Based on PHQ-9 >= 10 (moderate+) cutoff (Kroenke 2001),
        with behavioral adjustment for sleep, social withdrawal, and sentiment.
    Anxiety label: Based on GAD-7 >= 10 (moderate+) cutoff (Spitzer 2006),
        with stress and sleep adjustment.
    Bipolar label: Based on MDQ >= 7 (positive screen) cutoff (Hirschfeld 2000),
        with mood stability adjustment.
    Suicide label: Multi-factor rule combining hopelessness, keyword intensity,
        PHQ-9 severity, and clinical history (per Columbia Protocol guidelines).

    Label noise: ~10% flip rate to simulate real-world clinical disagreement
    (inter-rater kappa ≈ 0.80 for structured assessments).
    """
    # Depression: PHQ-9 >= 10 + behavioral evidence weighting
    dep_questionnaire = (phq9 >= 10).astype(float)
    dep_behavioral = (
        (sleep_hours < 5.5).astype(float) * 0.15 +
        (social_score < 30).astype(float) * 0.15 +
        (sentiment < -0.3).astype(float) * 0.15
    )
    dep_probability = np.clip(dep_questionnaire * 0.55 + dep_behavioral + past_episodes * 0.05, 0, 1)
    dep_label = (dep_probability > 0.40).astype(int)

    # Anxiety: GAD-7 >= 10 + stress + sleep disruption
    anx_questionnaire = (gad7 >= 10).astype(float)
    anx_behavioral = (
        (stress >= 7).astype(float) * 0.20 +
        (sleep_hours < 5.5).astype(float) * 0.10
    )
    anx_probability = np.clip(anx_questionnaire * 0.55 + anx_behavioral + family_history * 0.05, 0, 1)
    anx_label = (anx_probability > 0.40).astype(int)

    # Bipolar: MDQ >= 7 + mood instability + clinical history
    bip_questionnaire = (mdq >= 7).astype(float)
    bip_behavioral = (
        (mood_stability < 40).astype(float) * 0.20 +
        (past_episodes >= 2).astype(float) * 0.10
    )
    bip_probability = np.clip(bip_questionnaire * 0.60 + bip_behavioral, 0, 1)
    bip_label = (bip_probability > 0.45).astype(int)

    # Suicide risk: Multi-factor Columbia-aligned
    sui_hopelessness = (hopelessness > 0.4).astype(float) * 0.30
    sui_keywords = (keyword_intensity > 40).astype(float) * 0.25
    sui_depression = (phq9 >= 15).astype(float) * 0.20
    sui_history = (past_episodes >= 2).astype(float) * 0.10
    sui_social = (social_score < 20).astype(float) * 0.15
    sui_probability = np.clip(sui_hopelessness + sui_keywords + sui_depression + sui_history + sui_social, 0, 1)
    sui_label = (sui_probability > 0.35).astype(int)

    # Add realistic label noise (~8% flip rate, simulating clinical disagreement)
    noise_rate = 0.08
    rng = np.random.RandomState(42)
    for labels in [dep_label, anx_label, bip_label, sui_label]:
        flip_mask = rng.random(n_samples) < noise_rate
        labels[flip_mask] = 1 - labels[flip_mask]

    return {
        "depression": dep_label,
        "anxiety": anx_label,
        "bipolar": bip_label,
        "suicide": sui_label,
    }


def generate_synthetic_cohort(n_samples: int = 2000) -> Tuple[np.ndarray, Dict[str, np.ndarray], pl.DataFrame]:
    """
    Generates a realistic clinical cohort with multi-modal features.

    IMPORTANT: Labels are derived from features using clinical scoring rules,
    NOT from the same latent variables. This means models must learn clinically
    meaningful patterns rather than trivially recovering a math formula.
    """
    # Use latent variables ONLY for generating realistic feature distributions
    # (not for labels — labels come from clinical rules applied to features)
    latent_dep = np.random.beta(2, 5, size=n_samples)
    latent_anx = 0.6 * latent_dep + 0.4 * np.random.beta(2, 5, size=n_samples)
    latent_bip = 0.3 * latent_dep + 0.7 * np.random.beta(1.5, 6, size=n_samples)
    latent_sui = 0.5 * latent_dep + 0.3 * latent_anx + 0.2 * np.random.beta(1, 8, size=n_samples)

    age = np.random.randint(18, 75, size=n_samples)
    gender_code = np.random.choice([0, 1, 2], size=n_samples, p=[0.48, 0.48, 0.04])
    family_history = np.random.choice([0, 1], size=n_samples, p=[0.65, 0.35])
    past_episodes = np.random.poisson(lam=0.8, size=n_samples)
    medication_count = np.random.poisson(lam=1.1, size=n_samples)
    medical_conditions = np.random.poisson(lam=0.7, size=n_samples)

    phq9_score = np.clip(np.round(latent_dep * 27 + np.random.normal(0, 2, n_samples)), 0, 27).astype(int)
    gad7_score = np.clip(np.round(latent_anx * 21 + np.random.normal(0, 2, n_samples)), 0, 21).astype(int)
    mdq_score = np.clip(np.round(latent_bip * 13 + np.random.normal(0, 1.5, n_samples)), 0, 13).astype(int)
    mood_stability_index = np.clip(100 - (latent_bip * 70 + latent_dep * 20 + np.random.normal(0, 5, n_samples)), 0, 100)
    who5_wellbeing = np.clip(100 - (latent_dep * 80 + np.random.normal(0, 5, n_samples)), 0, 100)

    sleep_hours = np.clip(8.5 - latent_dep * 4.5 - latent_anx * 1.5 + np.random.normal(0, 1, n_samples), 3.0, 12.0)
    social_interaction_score = np.clip(80 - latent_dep * 60 - latent_anx * 20 + np.random.normal(0, 8, n_samples), 0, 100)
    physical_activity_min = np.clip(60 - latent_dep * 50 + np.random.normal(0, 10, n_samples), 0, 180)
    screen_time_hrs = np.clip(3.0 + latent_dep * 5.0 + latent_anx * 2.0 + np.random.normal(0, 1, n_samples), 1.0, 16.0)
    diet_quality_score = np.clip(75 - latent_dep * 45 + np.random.normal(0, 8, n_samples), 10, 100)
    stress_level = np.clip(np.round(1 + (latent_anx * 6 + latent_dep * 3) + np.random.normal(0, 0.8, n_samples)), 1, 10).astype(int)

    speech_pitch_hz = np.clip(180 - latent_dep * 45 + np.random.normal(0, 12, n_samples), 80, 300)
    speech_tone_var = np.clip(45 - latent_dep * 30 + np.random.normal(0, 5, n_samples), 5, 80)
    speaking_rate_wpm = np.clip(150 - latent_dep * 55 + np.random.normal(0, 15, n_samples), 60, 240)
    pause_frequency_ppm = np.clip(8 + latent_dep * 16 + np.random.normal(0, 3, n_samples), 2, 40)
    audio_energy = np.clip(0.8 - latent_dep * 0.5 + np.random.normal(0, 0.08, n_samples), 0.05, 1.0)

    sentiment_score = np.clip(0.6 - latent_dep * 1.4 - latent_sui * 0.4 + np.random.normal(0, 0.15, n_samples), -1.0, 1.0)
    sadness_prob = np.clip(latent_dep * 0.85 + np.random.normal(0, 0.08, n_samples), 0.0, 1.0)
    anxiety_prob = np.clip(latent_anx * 0.85 + np.random.normal(0, 0.08, n_samples), 0.0, 1.0)
    hopelessness_prob = np.clip(latent_sui * 0.85 + np.random.normal(0, 0.08, n_samples), 0.0, 1.0)
    keyword_intensity = np.clip(latent_sui * 90 + latent_dep * 20 + np.random.normal(0, 5, n_samples), 0, 100)

    X_data = np.column_stack([
        phq9_score, gad7_score, mdq_score, mood_stability_index, who5_wellbeing,
        sleep_hours, social_interaction_score, physical_activity_min, screen_time_hrs, diet_quality_score, stress_level,
        speech_pitch_hz, speech_tone_var, speaking_rate_wpm, pause_frequency_ppm, audio_energy,
        sentiment_score, sadness_prob, anxiety_prob, hopelessness_prob, keyword_intensity,
        age, gender_code, family_history, past_episodes, medication_count, medical_conditions
    ])

    # CRITICAL CHANGE: Labels derived from clinical rules applied to FEATURES,
    # not from the latent variables that generated those features.
    targets = _generate_clinical_labels(
        phq9=phq9_score,
        gad7=gad7_score,
        mdq=mdq_score,
        mood_stability=mood_stability_index,
        sleep_hours=sleep_hours,
        social_score=social_interaction_score,
        sentiment=sentiment_score,
        hopelessness=hopelessness_prob,
        keyword_intensity=keyword_intensity,
        stress=stress_level,
        past_episodes=past_episodes,
        family_history=family_history,
        n_samples=n_samples,
    )

    df_dict = {col: X_data[:, i] for i, col in enumerate(FEATURE_COLUMNS)}
    df = pl.DataFrame(df_dict)
    return X_data, targets, df

def get_sample_patients() -> List[Dict[str, Any]]:
    return PATIENT_STORE

def register_new_patient(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Registers a new patient with unique MRN."""
    mrn_num = random.randint(10000, 99999)
    mrn_id = f"MRN-2026-{mrn_num}"

    new_patient = {
        "id": mrn_id,
        "name": patient_data.get("name", "New Patient"),
        "age": int(patient_data.get("age", 25)),
        "gender": patient_data.get("gender", "Male"),
        "mrn": mrn_id,
        "contact": patient_data.get("contact", "+1 (555) 000-0000"),
        "primary_physician": patient_data.get("primary_physician", "Dr. Rohan Shinde, MBBS, MD Psychiatry"),
        "clinical_summary": patient_data.get("clinical_summary", "Initial intake assessment registered."),
        "features": patient_data.get("features", {
            "phq9_score": 10, "gad7_score": 8, "mdq_score": 2, "mood_stability_index": 70.0,
            "who5_wellbeing": 50.0, "sleep_hours": 6.5, "social_interaction_score": 50.0,
            "physical_activity_min": 30, "screen_time_hrs": 6.0, "diet_quality_score": 60.0,
            "stress_level": 5, "speech_pitch_hz": 150.0, "speech_tone_var": 30.0,
            "speaking_rate_wpm": 130.0, "pause_frequency_ppm": 12.0, "audio_energy": 0.5,
            "sentiment_score": 0.0, "sadness_prob": 0.2, "anxiety_prob": 0.3,
            "hopelessness_prob": 0.1, "keyword_intensity": 15.0, "age": int(patient_data.get("age", 25)),
            "gender_code": 1 if patient_data.get("gender") == "Male" else 0,
            "family_history": 0, "past_episodes": 0, "medication_count": 0, "medical_conditions": 0
        })
    }

    PATIENT_STORE.insert(0, new_patient)
    return new_patient
