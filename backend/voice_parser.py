"""
Voice Intake NLP Parser
Extracts clinical parameters from live voice/text transcriptions using:
1. Google Gemini 1.5/2.0 Flash (Intelligent intent detection & abrupt answer pivots)
2. Local VADER & enhanced context-aware heuristic fallback (Zero-API-Key offline safe)
"""

import re
from typing import Dict, Any, Optional
from ocr_parser import _analyze_sentiment_vader, _compute_clinical_keyword_analysis
from gemini_service import parse_conversational_response_gemini, get_gemini_api_key

# Common conversational abrupt/off-topic indicators
OFF_TOPIC_INDICATORS = {
    "birthday": ("Happy early birthday! Wishing you a great celebration!", "Roughly how many hours per night do you usually sleep?"),
    "tomorrow": ("Got it! To ensure we capture your health indicators accurately,", "Could you tell me how many hours you typically sleep?"),
    "food": ("Sounds good! To help me evaluate your well-being,", "Could you share your average hours of sleep or rest?"),
    "weather": ("Understood! On the health side,", "How have you been sleeping lately in terms of hours?"),
}

def parse_voice_transcript(transcript: str, context: str, question_text: str = "") -> Dict[str, Any]:
    """
    Parses a conversational patient response based on the question context.
    Contexts: 'sleep', 'mood', 'activity', 'stress', 'social', 'family', 'episodes', 'medications', 'conditions'
    """
    transcript_lower = transcript.lower().strip()
    
    # ---------------------------------------------------------
    # 1. Attempt Gemini Intelligence Parse (if key configured)
    # ---------------------------------------------------------
    if get_gemini_api_key():
        try:
            gemini_result = parse_conversational_response_gemini(transcript, context, question_text)
            if gemini_result and isinstance(gemini_result, dict):
                is_rel = gemini_result.get("is_relevant", True)
                reply = gemini_result.get("conversational_reply", "")
                requires_fu = gemini_result.get("requires_followup", False)
                suggested_fu = gemini_result.get("suggested_followup_question", "")
                params = gemini_result.get("extracted_parameters", {})

                # Compute baseline sentiment & keyword flags
                vader_scores = _analyze_sentiment_vader(transcript)
                keyword_analysis = _compute_clinical_keyword_analysis(transcript)
                
                features = {
                    "sentiment_score": gemini_result.get("sentiment_score", round(vader_scores["compound"], 4)),
                    "sadness_prob": gemini_result.get("sadness_prob", keyword_analysis["sadness_prob"]),
                    "anxiety_prob": gemini_result.get("anxiety_prob", 0.0),
                    "keyword_intensity": keyword_analysis["keyword_intensity"],
                    "hopelessness_prob": gemini_result.get("hopelessness_prob", min(1.0, keyword_analysis["keyword_intensity"] / 100 * 0.8)),
                    "suicide_risk_flag": keyword_analysis["suicide_risk_flag"]
                }
                
                # Merge extracted medical parameters
                for k, v in params.items():
                    if v is not None:
                        features[k] = v

                return {
                    "extracted_features": features,
                    "sentiment_breakdown": {
                        "sentiment_score": features["sentiment_score"],
                        "sadness_prob": features["sadness_prob"],
                        "anxiety_prob": features.get("anxiety_prob", 0.0),
                        "hopelessness_prob": features["hopelessness_prob"]
                    },
                    "transcript": transcript,
                    "context": context,
                    "is_relevant": is_rel,
                    "conversational_reply": reply,
                    "requires_followup": requires_fu,
                    "suggested_followup_question": suggested_fu,
                    "engine_used": "Google Gemini Intelligence Engine",
                    "keyword_flags": keyword_analysis["high_risk_terms_found"]
                }
        except Exception:
            pass  # Fall through seamlessly to local fallback

    # ---------------------------------------------------------
    # 2. Local Fallback with Smart Abrupt-Answer Heuristics
    # ---------------------------------------------------------
    vader_scores = _analyze_sentiment_vader(transcript)
    keyword_analysis = _compute_clinical_keyword_analysis(transcript)

    extracted = {
        "sentiment_score": round(vader_scores["compound"], 4),
        "sadness_prob": keyword_analysis["sadness_prob"],
        "keyword_intensity": keyword_analysis["keyword_intensity"],
        "hopelessness_prob": min(1.0, keyword_analysis["keyword_intensity"] / 100 * 0.8),
        "suicide_risk_flag": keyword_analysis["suicide_risk_flag"]
    }

    is_relevant = True
    conversational_reply = "Thank you for sharing. Response recorded."
    requires_followup = False
    suggested_followup = ""

    # Check for abrupt non-sequiturs (e.g. "my birthday is tomorrow")
    if "birthday" in transcript_lower or "bday" in transcript_lower:
        is_relevant = False
        requires_followup = True
        conversational_reply = "Happy early birthday! Wishing you a wonderful day! To complete your health profile, could you let me know roughly how many hours you sleep per night?"
        suggested_followup = "Roughly how many hours per night do you sleep?"
    elif context == "sleep" and any(word in transcript_lower for word in ["pizza", "lunch", "dinner", "movie", "weather"]):
        is_relevant = False
        requires_followup = True
        conversational_reply = "Understood! To make sure your evaluation is accurate, could you share your average hours of sleep?"
        suggested_followup = "About how many hours of sleep do you get?"

    # Context-specific extraction (with date/age filter guardrails)
    if is_relevant:
        if context == "sleep":
            # Guardrail: do NOT extract day dates like "15th" or "tomorrow" as sleep hours
            clean_text = re.sub(r'\b\d+(?:st|nd|rd|th)\b', '', transcript_lower)
            nums = re.findall(r'(\d+(?:\.\d+)?)', clean_text)
            if nums:
                valid_nums = [float(n) for n in nums if 2 <= float(n) <= 16]
                if valid_nums:
                    extracted["sleep_hours"] = valid_nums[0]
                    conversational_reply = f"Noted, approximately {valid_nums[0]} hours of sleep recorded."
            elif "hardly" in clean_text or "barely" in clean_text or "can't sleep" in clean_text or "insomnia" in clean_text:
                extracted["sleep_hours"] = 3.5
                conversational_reply = "Severe difficulty sleeping noted. We will factor this into your clinical care directives."

        elif context == "activity":
            if "zero" in transcript_lower or "none" in transcript_lower or "don't" in transcript_lower or "no exercise" in transcript_lower:
                extracted["physical_activity_min"] = 0.0
                conversational_reply = "Sedentary activity pattern noted."
            else:
                nums = re.findall(r'(\d+)', transcript_lower)
                if nums:
                    valid_nums = [float(n) for n in nums if 0 <= float(n) <= 300]
                    if valid_nums:
                        extracted["physical_activity_min"] = valid_nums[0]
                        conversational_reply = f"Recorded {valid_nums[0]} minutes of daily activity."

        elif context == "stress":
            nums = re.findall(r'(\d+)', transcript_lower)
            if nums:
                valid_nums = [int(n) for n in nums if 1 <= int(n) <= 10]
                if valid_nums:
                    extracted["stress_level"] = valid_nums[0]
                    conversational_reply = f"Stress level {valid_nums[0]}/10 recorded."
            if "stress_level" not in extracted:
                if extracted["sentiment_score"] < -0.5:
                    extracted["stress_level"] = 8
                elif extracted["sentiment_score"] < -0.2:
                    extracted["stress_level"] = 6
                else:
                    extracted["stress_level"] = 4
                conversational_reply = f"Stress estimated at {extracted['stress_level']}/10 based on sentiment."

        elif context == "social":
            if "alone" in transcript_lower or "isolated" in transcript_lower or "nobody" in transcript_lower:
                extracted["social_interaction_score"] = 15.0
                conversational_reply = "Feelings of social isolation noted."
            elif "friends" in transcript_lower or "family" in transcript_lower or "active" in transcript_lower:
                extracted["social_interaction_score"] = 70.0 if extracted["sentiment_score"] >= 0 else 40.0
                conversational_reply = "Social engagement score updated."

        elif context == "family":
            if any(w in transcript_lower for w in ["yes", "mother", "father", "brother", "sister", "family"]):
                extracted["family_history"] = 1
                conversational_reply = "Family history recorded."
            else:
                extracted["family_history"] = 0

        elif context == "episodes":
            if any(w in transcript_lower for w in ["yes", "past", "before", "recurrent"]):
                extracted["past_episodes"] = 1
                conversational_reply = "Previous episodes noted."
            else:
                extracted["past_episodes"] = 0

        elif context == "medications":
            if any(w in transcript_lower for w in ["none", "zero", "no"]):
                extracted["medication_count"] = 0
            else:
                nums = re.findall(r'(\d+)', transcript_lower)
                extracted["medication_count"] = int(nums[0]) if nums else 1

        elif context == "conditions":
            if any(w in transcript_lower for w in ["none", "no", "healthy"]):
                extracted["medical_conditions"] = 0
            else:
                extracted["medical_conditions"] = 1

        elif context in ("vitals", "bp"):
            bp_match = re.search(r'(\d{2,3})\s*(?:/|over)\s*(\d{2,3})', transcript_lower)
            if bp_match:
                sys_val = float(bp_match.group(1))
                dia_val = float(bp_match.group(2))
                extracted["systolic_bp"] = sys_val
                extracted["diastolic_bp"] = dia_val
                if sys_val >= 140 or dia_val >= 90:
                    extracted["bp_level"] = "Stage 2 High BP (>140/90)"
                elif sys_val >= 130 or dia_val >= 80:
                    extracted["bp_level"] = "Elevated BP (130-139/80-89)"
                else:
                    extracted["bp_level"] = "Normal (120/80 mmHg)"
                conversational_reply = f"Blood pressure noted at {int(sys_val)}/{int(dia_val)} mmHg."
            elif any(w in transcript_lower for w in ["high", "elevated", "palpitations", "racing", "tightness", "headache", "pounding"]):
                extracted["systolic_bp"] = 142.0
                extracted["diastolic_bp"] = 92.0
                extracted["bp_level"] = "Elevated Tension / High BP"
                conversational_reply = "Elevated blood pressure sensation and somatic tension noted."
            elif any(w in transcript_lower for w in ["low", "sluggish", "dizzy", "lightheaded"]):
                extracted["systolic_bp"] = 98.0
                extracted["diastolic_bp"] = 62.0
                extracted["bp_level"] = "Low BP / Hypotensive"
                conversational_reply = "Low blood pressure and lightheadedness noted."
            else:
                extracted["systolic_bp"] = 120.0
                extracted["diastolic_bp"] = 80.0
                extracted["bp_level"] = "Normal (120/80 mmHg)"
                conversational_reply = "Cardiovascular vital signs noted in normal baseline range."

    return {
        "extracted_features": extracted,
        "sentiment_breakdown": {
            "sentiment_score": extracted["sentiment_score"],
            "sadness_prob": extracted["sadness_prob"],
            "anxiety_prob": extracted.get("anxiety_prob", 0.0),
            "hopelessness_prob": extracted["hopelessness_prob"]
        },
        "transcript": transcript,
        "context": context,
        "is_relevant": is_relevant,
        "conversational_reply": conversational_reply,
        "requires_followup": requires_followup,
        "suggested_followup_question": suggested_followup,
        "engine_used": "Local Explainable Clinical Heuristic Engine (Offline Safe)",
        "keyword_flags": keyword_analysis["high_risk_terms_found"]
    }
