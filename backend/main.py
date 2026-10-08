"""
FastAPI Server - Mental Health Risk Detection & Preventive Care Platform Backend API.
Provides patient registration, OCR document scanning, live speech audio feature extraction,
multi-modal risk scoring, Explainable AI (SHAP/LIME), FHIR R4 JSON export, Multi-Doctor Co-Signing,
and Gamified Patient Quiz Intake.

IMPORTANT CHANGES (v2.0 — Clinical Integrity Rewrite):
  - Quiz Intake now uses validated clinical scoring rules (PHQ-9/GAD-7).
  - Validation layer checks for completeness and contradictions.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

from data_generator import get_sample_patients, register_new_patient
from feature_engineering import compute_derived_features, generate_longitudinal_trend
from ml_engine import ml_engine
from xai_engine import generate_patient_xai
from preventive_engine import generate_preventive_care_plan
from clinical_llm import generate_clinical_summary
from ocr_parser import parse_medical_document_text
from doctor_signature import generate_doctor_verification_certificate
from fhir_exporter import generate_fhir_r4_bundle
from multi_doctor import add_doctor_cosignature, get_patient_consultation_logs
from clinical_scoring import estimate_phq9_from_behavioral, estimate_gad7_from_behavioral
from clinical_validator import validate_patient_input
from voice_parser import parse_voice_transcript

app = FastAPI(
    title="Mental Health Risk Detection & Preventive Care Platform API",
    description="AI-Powered Clinical Decision Support System API for Hospitals",
    version="2.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientRegistrationInput(BaseModel):
    name: str = Field(..., example="Vedant")
    age: int = Field(default=21, ge=1, le=120)
    gender: str = Field(default="Male")
    contact: Optional[str] = "+1 (555) 000-0000"
    primary_physician: Optional[str] = "Dr. Rohan Shinde, MBBS, MD Psychiatry"
    clinical_summary: Optional[str] = "New patient registered for clinical screening."

class DocumentScanInput(BaseModel):
    document_text: str = Field(..., description="Raw text extracted from medical lab report or questionnaire document")

class CosignInput(BaseModel):
    patient_id: str
    doctor_name: Optional[str] = "Dr. Amanda Vance, MD"
    comments: Optional[str] = "Reviewed and approved clinical AI evaluation and preventive plan."

class VoiceIntakeInput(BaseModel):
    transcript: str = Field(..., description="Transcribed voice text")
    context: str = Field(..., description="Context of the question asked (sleep, mood, activity, stress, social)")

class QuizIntakeInput(BaseModel):
    patient_id: Optional[str] = "MRN-2026-1023"
    life_sentence: str = Field(default="I feel fatigued and overwhelmed by work lately.", description="Write a sentence about your life right now")
    sleep_habit: str = Field(default="4-5 hours irregular sleep", description="Sleep quality description")
    sleep_hours: float = Field(default=4.5, ge=2.0, le=14.0)
    social_balance: str = Field(default="High social media, minimal face to face", description="Social activity vs screen time")
    screen_hrs: float = Field(default=9.5, ge=0.5, le=18.0)
    social_score: float = Field(default=25.0, ge=0.0, le=100.0)
    exercise_hobbies: str = Field(default="Light walking 15 mins daily", description="Best choices & daily habits")
    activity_min: float = Field(default=15.0, ge=0.0, le=240.0)
    stress_level: int = Field(default=8, ge=1, le=10)

class PatientFeaturesInput(BaseModel):
    id: Optional[str] = "MRN-2026-1023"
    name: Optional[str] = "Patient Intake"
    age: int = Field(default=21, ge=1, le=120)
    gender: str = Field(default="Male")
    phq9_score: float = Field(default=22.0, ge=0.0, le=27.0)
    gad7_score: float = Field(default=14.0, ge=0.0, le=21.0)
    mdq_score: float = Field(default=3.0, ge=0.0, le=13.0)
    mood_stability_index: float = Field(default=72.0, ge=0.0, le=100.0)
    who5_wellbeing: float = Field(default=18.0, ge=0.0, le=100.0)
    sleep_hours: float = Field(default=4.5, ge=0.0, le=24.0)
    social_interaction_score: float = Field(default=18.0, ge=0.0, le=100.0)
    physical_activity_min: float = Field(default=15.0, ge=0.0, le=300.0)
    screen_time_hrs: float = Field(default=9.5, ge=0.0, le=24.0)
    diet_quality_score: float = Field(default=35.0, ge=0.0, le=100.0)
    stress_level: int = Field(default=8, ge=1, le=10)
    speech_pitch_hz: float = Field(default=125.0, ge=50.0, le=400.0)
    speech_tone_var: float = Field(default=18.0, ge=0.0, le=100.0)
    speaking_rate_wpm: float = Field(default=92.0, ge=30.0, le=300.0)
    pause_frequency_ppm: float = Field(default=24.0, ge=0.0, le=60.0)
    audio_energy: float = Field(default=0.22, ge=0.0, le=1.0)
    sentiment_score: float = Field(default=-0.68, ge=-1.0, le=1.0)
    sadness_prob: float = Field(default=0.84, ge=0.0, le=1.0)
    anxiety_prob: float = Field(default=0.62, ge=0.0, le=1.0)
    hopelessness_prob: float = Field(default=0.58, ge=0.0, le=1.0)
    keyword_intensity: float = Field(default=65.0, ge=0.0, le=100.0)
    family_history: int = Field(default=1, ge=0, le=1)
    past_episodes: int = Field(default=2, ge=0, le=10)
    medication_count: int = Field(default=1, ge=0, le=10)
    medical_conditions: int = Field(default=0, ge=0, le=10)

@app.on_event("startup")
def startup_event():
    if not ml_engine.is_trained:
        ml_engine.train_pipeline(n_samples=2000)

@app.get("/")
def get_root():
    return {
        "status": "ONLINE",
        "system": "Mental Health Risk Detection & Preventive Care Platform",
        "domain": "Clinical Decision Support System for Hospitals",
        "version": "2.2.0",
        "ml_engine_status": "TRAINED" if ml_engine.is_trained else "INITIALIZING"
    }

@app.get("/api/patients")
def get_patients():
    return {"patients": get_sample_patients()}

@app.post("/api/patients/register")
def register_patient(payload: PatientRegistrationInput):
    new_p = register_new_patient(payload.model_dump() if hasattr(payload, 'model_dump') else payload.dict())
    return {
        "status": "REGISTERED_SUCCESSFULLY",
        "patient": new_p
    }

@app.post("/api/scan-document")
def scan_document(payload: DocumentScanInput):
    result = parse_medical_document_text(payload.document_text)
    return result

@app.post("/api/voice-intake")
def voice_intake(payload: VoiceIntakeInput):
    """Processes a single conversational response during live voice intake."""
    result = parse_voice_transcript(payload.transcript, payload.context)
    return result

@app.post("/api/quiz-intake")
def quiz_intake(payload: QuizIntakeInput):
    """Converts gamified patient quiz answers into clinical multi-modal parameters and runs inference."""
    ocr_result = parse_medical_document_text(payload.life_sentence + " " + payload.exercise_hobbies)
    extracted = ocr_result.get("extracted_features", {})
    sentiment = extracted.get("sentiment_score", 0.0)
    sadness_prob = extracted.get("sadness_prob", 0.0)
    keyword_intensity = extracted.get("keyword_intensity", 0.0)
    hopelessness_prob = min(1.0, keyword_intensity / 100 * 0.8)

    # 1. Use clinical scoring module to derive PHQ-9 and GAD-7 legitimately
    phq9_est = estimate_phq9_from_behavioral(
        sleep_hours=payload.sleep_hours,
        stress_level=payload.stress_level,
        social_score=payload.social_score,
        activity_min=payload.activity_min,
        sentiment_score=sentiment,
        screen_hrs=payload.screen_hrs,
        sadness_prob=sadness_prob,
        hopelessness_prob=hopelessness_prob,
        keyword_intensity=keyword_intensity
    )

    gad7_est = estimate_gad7_from_behavioral(
        stress_level=payload.stress_level,
        sleep_hours=payload.sleep_hours,
        sentiment_score=sentiment,
        anxiety_prob=extracted.get("anxiety_prob", 0.0),
        social_score=payload.social_score,
        screen_hrs=payload.screen_hrs
    )

    # Parallel Feature Fusion
    features = {
        "phq9_score": phq9_est["estimated_score"],
        "gad7_score": gad7_est["estimated_score"],
        "mdq_score": 3.0, # Baseline
        "who5_wellbeing": 100 - phq9_est["estimated_score"] * 3, # Correlated estimate
        "sleep_hours": payload.sleep_hours,
        "social_interaction_score": payload.social_score,
        "physical_activity_min": payload.activity_min,
        "screen_time_hrs": payload.screen_hrs,
        "stress_level": payload.stress_level,
        "speech_pitch_hz": 135.0,
        "speaking_rate_wpm": 110.0,
        "pause_frequency_ppm": 16.0,
        "speech_tone_var": 25.0,
        "sentiment_score": sentiment,
        "sadness_prob": sadness_prob,
        "hopelessness_prob": hopelessness_prob,
        "keyword_intensity": keyword_intensity,
        "suicide_risk_flag": ocr_result.get("keyword_analysis", {}).get("suicide_risk_flag", False)
    }

    # Validate inputs
    validation = validate_patient_input(features)

    prediction = ml_engine.predict_patient_risk(features)
    
    # Adjust confidence based on input validation quality
    adjusted_confidence = round(prediction["confidence_score"] * validation["confidence_adjustment"], 1)
    prediction["confidence_score"] = adjusted_confidence

    xai = generate_patient_xai(features)
    care_plan = generate_preventive_care_plan(features, prediction)

    return {
        "quiz_status": "QUIZ_PARALLEL_DATA_FILLED",
        "extracted_quiz_features": features,
        "validation_report": validation,
        "risk_scores": prediction["risk_scores"],
        "risk_categories": prediction["risk_categories"],
        "confidence_score": prediction["confidence_score"],
        "top_contributing_factors": xai["top_contributing_factors"],
        "recommended_actions": care_plan["recommended_actions"]
    }

@app.post("/api/predict")
def predict_risk(payload: PatientFeaturesInput):
    feat_dict = payload.model_dump() if hasattr(payload, 'model_dump') else payload.dict()
    g_map = {"Female": 0, "Male": 1, "Other": 2}
    feat_dict["gender_code"] = g_map.get(payload.gender, 1)

    validation = validate_patient_input(feat_dict)
    prediction = ml_engine.predict_patient_risk(feat_dict)
    
    # Adjust confidence based on data quality
    adjusted_confidence = round(prediction["confidence_score"] * validation["confidence_adjustment"], 1)
    prediction["confidence_score"] = adjusted_confidence

    trend = generate_longitudinal_trend(prediction["risk_scores"], weeks=12)

    return {
        "patient_id": payload.id,
        "patient_name": payload.name,
        "age": payload.age,
        "gender": payload.gender,
        "validation_report": validation,
        "risk_scores": prediction["risk_scores"],
        "risk_categories": prediction["risk_categories"],
        "derived_indices": prediction["derived_indices"],
        "alert_triggered": prediction["alert_triggered"],
        "alert_level": prediction["alert_level"],
        "confidence_score": prediction["confidence_score"],
        "historical_trend": trend
    }

@app.post("/api/explain")
def explain_risk(payload: PatientFeaturesInput):
    feat_dict = payload.model_dump() if hasattr(payload, 'model_dump') else payload.dict()
    g_map = {"Female": 0, "Male": 1, "Other": 2}
    feat_dict["gender_code"] = g_map.get(payload.gender, 1)

    xai = generate_patient_xai(feat_dict, target_condition="depression")
    return xai

@app.post("/api/doctor-report")
def get_doctor_signed_report(payload: PatientFeaturesInput):
    feat_dict = payload.model_dump() if hasattr(payload, 'model_dump') else payload.dict()
    g_map = {"Female": 0, "Male": 1, "Other": 2}
    feat_dict["gender_code"] = g_map.get(payload.gender, 1)

    validation = validate_patient_input(feat_dict)
    prediction = ml_engine.predict_patient_risk(feat_dict)
    prediction["confidence_score"] = round(prediction["confidence_score"] * validation["confidence_adjustment"], 1)

    xai = generate_patient_xai(feat_dict)
    care_plan = generate_preventive_care_plan(feat_dict, prediction)
    summaries = generate_clinical_summary(payload.name, payload.age, payload.gender, prediction, xai)
    
    cert = generate_doctor_verification_certificate(payload.id, payload.name, prediction["risk_scores"])
    consultation_logs = get_patient_consultation_logs(payload.id)

    return {
        "report_id": f"REP-{payload.id}",
        "patient_id": payload.id,
        "patient_name": payload.name,
        "age": payload.age,
        "gender": payload.gender,
        "validation_report": validation,
        "risk_scores": prediction["risk_scores"],
        "risk_categories": prediction["risk_categories"],
        "top_contributing_factors": xai["top_contributing_factors"],
        "preventive_actions": care_plan["recommended_actions"],
        "detailed_action_plan": care_plan["detailed_action_plan"],
        "followup_timeline": care_plan["followup_timeline"],
        "doctor_note": summaries["doctor_note"],
        "doctor_certificate": cert,
        "consultation_logs": consultation_logs
    }

@app.post("/api/export-fhir")
def export_fhir(payload: PatientFeaturesInput):
    """Exports patient assessment record in standard FHIR R4 JSON format."""
    feat_dict = payload.model_dump() if hasattr(payload, 'model_dump') else payload.dict()
    prediction = ml_engine.predict_patient_risk(feat_dict)
    cert = generate_doctor_verification_certificate(payload.id, payload.name, prediction["risk_scores"])
    bundle = generate_fhir_r4_bundle(payload.id, payload.name, payload.age, payload.gender, prediction["risk_scores"], feat_dict, cert)
    return bundle

@app.post("/api/multi-doctor/cosign")
def cosign_doctor_report(payload: CosignInput):
    """Adds second opinion doctor co-signature and review comment."""
    res = add_doctor_cosignature(payload.patient_id, {"name": payload.doctor_name, "qualification": "MD Psychiatry, Chief of Clinical Consultation", "hospital": "Memorial Neuropsychiatric Hospital", "license_number": "MP-91024-IN"}, payload.comments)
    return res

@app.get("/api/metrics")
def get_model_metrics():
    if not ml_engine.is_trained:
        ml_engine.train_pipeline()
    return {"metrics": ml_engine.metrics}

@app.get("/api/datasets")
def get_datasets_info():
    return {
        "datasets": [
            {"name": "PHQ-9 Dataset (Kaggle)", "purpose": "Depression Severity", "modality": "Questionnaire", "sample_count": 5000},
            {"name": "GAD-7 Dataset (Kaggle)", "purpose": "Anxiety Severity", "modality": "Questionnaire", "sample_count": 4200},
            {"name": "MDQ Dataset", "purpose": "Bipolar Screening", "modality": "Questionnaire", "sample_count": 3100},
            {"name": "DAIC-WOZ", "purpose": "Audio + Transcripts + Depression Labels", "modality": "Speech + Text", "sample_count": 142},
            {"name": "GoEmotions (Google)", "purpose": "28 Fine-grained Emotions", "modality": "Text NLP", "sample_count": 58000},
            {"name": "Sentiment140", "purpose": "Sentiment Analysis Baseline", "modality": "Text NLP", "sample_count": 1600000},
            {"name": "Reddit Depression Dataset", "purpose": "Depression Text Indicators", "modality": "Text NLP", "sample_count": 7500},
            {"name": "NHANES Dataset", "purpose": "Demographics & Clinical Data", "modality": "Tabular", "sample_count": 9254},
            {"name": "Suicide Detection Dataset", "purpose": "Suicide Risk Text Screening", "modality": "Text NLP", "sample_count": 232074}
        ]
    }
