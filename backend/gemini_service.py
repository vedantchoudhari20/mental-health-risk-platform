"""
Gemini Intelligence Service Module
Integrates Google Gemini 1.5/2.0 Flash via standard REST API for:
1. Intelligent abrupt/off-topic conversational answer handling with empathetic pivots.
2. Nuanced psychiatric sentiment & cognitive distortion extraction.
3. Hospital-grade psychiatric SOAP progress notes.
4. Seamless offline-first graceful fallback to local clinical heuristics.
"""

import os
import json
import re
from typing import Dict, Any, Optional
import requests

# Persistent or environment API key
_GEMINI_API_KEY: Optional[str] = os.environ.get("GEMINI_API_KEY", None)

ENV_FILE_PATH = os.path.join(os.path.dirname(__file__), ".env")
if not _GEMINI_API_KEY and os.path.exists(ENV_FILE_PATH):
    try:
        with open(ENV_FILE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    _GEMINI_API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    except Exception:
        pass


def get_gemini_api_key() -> Optional[str]:
    """Returns the currently active Gemini API key if present."""
    global _GEMINI_API_KEY
    return _GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", None)


def set_gemini_api_key(key: str, persist: bool = True) -> Dict[str, Any]:
    """Sets and validates a new Gemini API key, optionally saving to .env."""
    global _GEMINI_API_KEY
    cleaned_key = key.strip() if key else ""
    if not cleaned_key:
        _GEMINI_API_KEY = None
        return {"status": "CLEARED", "message": "Gemini API key cleared. Operating in local mode."}
    
    # Test key with a ping
    test_result = test_gemini_connection(cleaned_key)
    if not test_result.get("success"):
        return {"status": "ERROR", "message": test_result.get("error", "Failed to validate key with Google Gemini API.")}

    _GEMINI_API_KEY = cleaned_key
    os.environ["GEMINI_API_KEY"] = cleaned_key

    if persist:
        try:
            with open(ENV_FILE_PATH, "w", encoding="utf-8") as f:
                f.write(f"GEMINI_API_KEY={cleaned_key}\n")
        except Exception:
            pass

    return {
        "status": "CONNECTED",
        "model": test_result.get("model", "gemini-1.5-flash"),
        "message": "Gemini API key validated and activated successfully!"
    }


# Model priority chain: newest free model first, then fallbacks
GEMINI_MODELS = ["gemini-3.8-flash", "gemini-2.0-flash", "gemini-1.5-flash"]


def test_gemini_connection(api_key: str) -> Dict[str, Any]:
    """Pings Gemini API to check key validity across available models."""
    if not api_key:
        return {"success": False, "error": "No API key provided."}
    
    for model in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": "Hello, respond with OK."}]}],
            "generationConfig": {"maxOutputTokens": 10}
        }
        try:
            resp = requests.post(url, json=payload, timeout=12)
            if resp.status_code == 200:
                return {"success": True, "model": model}
            elif resp.status_code in (400, 403):
                data = resp.json()
                msg = data.get("error", {}).get("message", f"HTTP {resp.status_code}")
                return {"success": False, "error": f"Gemini API Error: {msg}"}
            # 404 means model deprecated, try next
        except Exception:
            continue
    return {"success": False, "error": "Network timeout or unable to reach Google Gemini endpoint."}


def _call_gemini_json(prompt: str, system_instruction: str = "") -> Optional[Dict[str, Any]]:
    """Helper to call Gemini API and parse structured JSON response."""
    key = get_gemini_api_key()
    if not key:
        return None

    for model in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        
        contents = []
        if system_instruction:
            contents.append({"role": "user", "parts": [{"text": f"SYSTEM INSTRUCTION: {system_instruction}\n\nUSER PROMPT: {prompt}"}]})
        else:
            contents.append({"role": "user", "parts": [{"text": prompt}]})

        # Note: gemini-3.8-flash is a thinking model; do NOT use responseMimeType application/json
        # as it conflicts with thinking token output. Parse JSON from text instead.
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1500
            }
        }

        try:
            resp = requests.post(url, json=payload, timeout=18)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    content_parts = candidates[0].get("content", {}).get("parts", [])
                    # Collect all text parts (skip thought/thinking parts)
                    text_out = ""
                    for part in content_parts:
                        txt = part.get("text", "")
                        if txt:
                            text_out += txt
                    text_out = text_out.strip()
                    if not text_out:
                        continue  # thinking model returned empty, try next
                    # Extract JSON from markdown code fence if present
                    if "```json" in text_out:
                        text_out = text_out.split("```json", 1)[1]
                        text_out = text_out.split("```", 1)[0]
                    elif "```" in text_out:
                        text_out = text_out.split("```", 1)[1]
                        text_out = text_out.split("```", 1)[0]
                    # Find first { ... } block
                    start = text_out.find("{")
                    end = text_out.rfind("}")
                    if start != -1 and end != -1:
                        text_out = text_out[start:end+1]
                    return json.loads(text_out.strip())
            elif resp.status_code == 404:
                continue  # model deprecated, try next
        except Exception:
            continue
    return None


def parse_conversational_response_gemini(transcript: str, context: str, question_text: str = "") -> Optional[Dict[str, Any]]:
    """
    Intelligently analyzes patient's spoken or typed answer to clinical questions.
    Detects if the answer is:
    - Direct (e.g., "5 hours, broken sleep")
    - Indirect / Emotional (e.g., "My mind won't let me shut down")
    - Abrupt / Off-topic / Non-sequitur (e.g., "My birthday is tomorrow", "I had pizza for lunch", "Why are you asking?")
    
    Returns structured parameters and an empathetic conversational reply/pivot!
    """
    system_inst = """
You are an expert, compassionate clinical psychiatric AI interviewer assisting a doctor in an EHR system.
A patient was asked a specific medical/lifestyle question.
Your job is to parse their response, detect relevance, handle abrupt or off-topic non-sequiturs empathetically, and extract parameters.

CRITICAL RULES:
1. If the user answered off-topic or abruptly (e.g., asked about sleep, but user talks about birthdays, lunch, movies, weather, or random topics):
   - Set "is_relevant": false
   - Set "requires_followup": true
   - In "conversational_reply", warmly acknowledge what they said (e.g., "Happy early birthday! Wishing you a wonderful celebration!") and then gently pivot back to the medical question (e.g., "To make sure you're feeling your best, could you also let me know roughly how many hours of sleep you get at night?").
   - DO NOT extract numbers that refer to dates, birthdays, or ages as sleep hours or activity minutes! (e.g., "my birthday is on 15th" must NOT be extracted as 15 hours of sleep!).

2. If the user gave a relevant direct or indirect answer:
   - Set "is_relevant": true
   - Set "requires_followup": false
   - In "conversational_reply", briefly acknowledge their answer in a supportive medical tone.
   - Extract numerical parameters appropriate to the context:
     * sleep: "sleep_hours" (float 0.0 - 24.0)
     * activity: "physical_activity_min" (float minutes/day)
     * stress: "stress_level" (int 1 - 10)
     * vitals / bp: "systolic_bp" (float, e.g. 120.0), "diastolic_bp" (float, e.g. 80.0), "bp_level" (string, e.g. "Normal (120/80 mmHg)", "Elevated BP (130-139/80-89)", "Stage 2 High BP (>140/90)", "Low BP / Hypotensive")
     * social: "social_interaction_score" (float 0 - 100)
     * family: "family_history" (int 0 or 1)
     * episodes: "past_episodes" (int 0 or 1)
     * medications: "medication_count" (int count)
     * conditions: "medical_conditions" (int count)

Output strict JSON schema:
{
  "is_relevant": boolean,
  "detected_topic": string,
  "conversational_reply": string,
  "requires_followup": boolean,
  "suggested_followup_question": string,
  "sentiment_score": float (-1.0 to 1.0),
  "sadness_prob": float (0.0 to 1.0),
  "anxiety_prob": float (0.0 to 1.0),
  "hopelessness_prob": float (0.0 to 1.0),
  "extracted_parameters": object
}
"""

    prompt = f"""
Clinical Question Asked: "{question_text or context}"
Question Context Category: "{context}"
Patient's Answer: "{transcript}"

Analyze this answer and respond in the specified JSON format.
"""

    result = _call_gemini_json(prompt, system_instruction=system_inst)
    return result


def analyze_narrative_with_gemini(narrative_text: str) -> Optional[Dict[str, Any]]:
    """
    Performs deep clinical psychiatric sentiment analysis on open-ended text (e.g. Life Sentence in Gamified Quiz).
    """
    system_inst = """
You are a senior clinical neuropsychiatrist assessing open-ended patient self-narratives.
Evaluate the patient's text for:
- Clinical sentiment (-1.0 to +1.0)
- Probability of sadness, anxiety, hopelessness (0.0 to 1.0)
- Suicidal ideation / crisis flags (boolean)
- Cognitive distortions identified (e.g., "catastrophizing", "all-or-nothing thinking", "emotional reasoning", or "none")
- Key clinical themes (array of strings)
- Empathetic clinician reflection (1-2 sentences of psychoeducational feedback)

Output strict JSON schema:
{
  "sentiment_score": float,
  "sadness_prob": float,
  "anxiety_prob": float,
  "hopelessness_prob": float,
  "suicide_risk_flag": boolean,
  "cognitive_distortions": [string],
  "clinical_themes": [string],
  "empathetic_reflection": string
}
"""
    prompt = f"Patient Narrative: \"{narrative_text}\""
    return _call_gemini_json(prompt, system_instruction=system_inst)


def generate_gemini_soap_note(patient_name: str, age: int, gender: str, risk_scores: Dict[str, float], risk_categories: Dict[str, str], top_factors: list, qualitative_notes: str = "") -> Optional[Dict[str, str]]:
    """
    Generates a hospital-grade psychiatric SOAP (Subjective, Objective, Assessment, Plan) progress note.
    """
    system_inst = """
You are Dr. Rohan Shinde, MBBS, MD Psychiatry, Chief of Clinical Neuropsychiatry at Memorial Neuropsychiatric Hospital.
Write an authentic, professional, comprehensive SOAP Progress Note for the patient's official electronic medical record.
Structure the note strictly with:
- SUBJECTIVE: Chief complaint, patient symptoms, sleep/appetite/energy narrative.
- OBJECTIVE: Multi-modal AI risk assessment scores, SHAP top contributing factors, speech acoustic observations.
- ASSESSMENT: Clinical psychiatric risk formulation (Depression, GAD, Bipolar features, Crisis alert).
- PLAN: Evidence-based pharmacological/psychotherapeutic directives (CBT), sleep hygiene, safety precautions, follow-up timeline.

Output strict JSON schema:
{
  "soap_note": string,
  "patient_explanation": string,
  "clinical_summary": string
}
"""
    prompt = f"""
Patient: {patient_name}, {age} y/o {gender}
Risk Scores: {json.dumps(risk_scores)}
Risk Categories: {json.dumps(risk_categories)}
Top SHAP Risk Drivers: {json.dumps(top_factors)}
Patient Self-Report / Qualitative History: "{qualitative_notes}"
"""
    return _call_gemini_json(prompt, system_instruction=system_inst)
