import requests
import json

url = "http://127.0.0.1:8000/api/predict"

features = {
    "id": "MRN-2026-9999",
    "name": "Jane Doe",
    "age": 28,
    "gender": "Female",
    "phq9_score": 15.0,  # Moderate-severe
    "gad7_score": 14.0,  # Moderate
    "mdq_score": 2.0,    # Low bipolar risk
    "mood_stability_index": 70.0,
    "who5_wellbeing": 40.0,
    "sleep_hours": 4.5,
    "social_interaction_score": 25.0,
    "physical_activity_min": 10.0,
    "screen_time_hrs": 10.5,
    "diet_quality_score": 40.0,
    "stress_level": 8,
    "speech_pitch_hz": 120.0,
    "speech_tone_var": 20.0,
    "speaking_rate_wpm": 90.0,
    "pause_frequency_ppm": 25.0,
    "audio_energy": 0.3,
    "sentiment_score": -0.75,
    "sadness_prob": 0.8,
    "anxiety_prob": 0.7,
    "hopelessness_prob": 0.6,
    "keyword_intensity": 55.0,
    "family_history": 1,
    "past_episodes": 1,
    "medication_count": 0,
    "medical_conditions": 0
}

print("=== CLINICAL ML PREDICTION TEST ===\n")
resp = requests.post(url, json=features)
if resp.status_code == 200:
    data = resp.json()
    print("-> Risk Scores:", json.dumps(data["risk_scores"], indent=2))
    print("-> Risk Categories:", json.dumps(data["risk_categories"], indent=2))
    print("-> Alert Level:", data["alert_level"])
    print("-> Confidence Score:", data["confidence_score"])
    print("-> Validation Report Quality:", data["validation_report"]["quality_grade"])
else:
    print("Error:", resp.text)
