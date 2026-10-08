"""
Voice Intake NLP Parser
Extracts clinical parameters from live voice transcriptions using VADER and Regex.
"""

import re
from typing import Dict, Any
from ocr_parser import _analyze_sentiment_vader, _compute_clinical_keyword_analysis

def parse_voice_transcript(transcript: str, context: str) -> Dict[str, Any]:
    """
    Parses a single voice response based on the context of the question asked.
    Contexts: 'sleep', 'mood', 'activity', 'stress', 'social'
    """
    transcript_lower = transcript.lower()
    extracted = {}
    
    # 1. Analyze sentiment and distress keywords universally
    vader_scores = _analyze_sentiment_vader(transcript)
    keyword_analysis = _compute_clinical_keyword_analysis(transcript)
    
    extracted["sentiment_score"] = round(vader_scores["compound"], 4)
    extracted["sadness_prob"] = keyword_analysis["sadness_prob"]
    extracted["keyword_intensity"] = keyword_analysis["keyword_intensity"]
    extracted["hopelessness_prob"] = min(1.0, keyword_analysis["keyword_intensity"] / 100 * 0.8)
    extracted["suicide_risk_flag"] = keyword_analysis["suicide_risk_flag"]

    # 2. Context-specific extraction
    if context == "sleep":
        # Extract hours (e.g., "about 4 hours", "5 or 6 hours", "I sleep 8.5 hours")
        nums = re.findall(r'(\d+(?:\.\d+)?)', transcript_lower)
        if nums:
            # Pick the most likely number representing hours (between 0 and 24)
            valid_nums = [float(n) for n in nums if 2 <= float(n) <= 16]
            if valid_nums:
                extracted["sleep_hours"] = valid_nums[0]
                
    elif context == "activity":
        # Extract minutes (e.g., "walk for 30 minutes", "15 mins", "zero")
        if "zero" in transcript_lower or "none" in transcript_lower or "don't" in transcript_lower:
            extracted["physical_activity_min"] = 0.0
        else:
            nums = re.findall(r'(\d+)', transcript_lower)
            if nums:
                valid_nums = [float(n) for n in nums if 0 <= float(n) <= 300]
                if valid_nums:
                    extracted["physical_activity_min"] = valid_nums[0]
                    
    elif context == "stress":
        # Extract stress level 1-10
        nums = re.findall(r'(\d+)', transcript_lower)
        if nums:
            valid_nums = [int(n) for n in nums if 1 <= int(n) <= 10]
            if valid_nums:
                extracted["stress_level"] = valid_nums[0]
        
        # Adjust stress based on sentiment if no number found
        if "stress_level" not in extracted:
            if extracted["sentiment_score"] < -0.5:
                extracted["stress_level"] = 8
            elif extracted["sentiment_score"] < -0.2:
                extracted["stress_level"] = 6
            elif extracted["sentiment_score"] > 0.2:
                extracted["stress_level"] = 3
                
    elif context == "social":
        # Social score based on keywords and sentiment
        if "alone" in transcript_lower or "isolated" in transcript_lower or "nobody" in transcript_lower:
            extracted["social_interaction_score"] = 15.0
        elif "friends" in transcript_lower or "family" in transcript_lower or "people" in transcript_lower:
            if extracted["sentiment_score"] > 0:
                extracted["social_interaction_score"] = 75.0
            else:
                extracted["social_interaction_score"] = 40.0
                
    elif context == "family":
        if "yes" in transcript_lower or "they have" in transcript_lower or "mother" in transcript_lower or "father" in transcript_lower or "brother" in transcript_lower or "sister" in transcript_lower:
            extracted["family_history"] = 1
        elif "no" in transcript_lower or "none" in transcript_lower:
            extracted["family_history"] = 0
            
    elif context == "episodes":
        if "never" in transcript_lower or "no" in transcript_lower or "first time" in transcript_lower:
            extracted["past_episodes"] = 0
        elif "yes" in transcript_lower or "before" in transcript_lower or "in the past" in transcript_lower or "always" in transcript_lower:
            extracted["past_episodes"] = 1
            
    elif context == "medications":
        if "none" in transcript_lower or "zero" in transcript_lower or "no" in transcript_lower:
            extracted["medication_count"] = 0
        else:
            nums = re.findall(r'(\d+)', transcript_lower)
            if nums:
                valid_nums = [int(n) for n in nums if 0 <= int(n) <= 15]
                if valid_nums:
                    extracted["medication_count"] = valid_nums[0]
            elif "couple" in transcript_lower or "few" in transcript_lower or "two" in transcript_lower:
                extracted["medication_count"] = 2
            elif "one" in transcript_lower or "a pill" in transcript_lower or "just this" in transcript_lower:
                extracted["medication_count"] = 1
                
    elif context == "conditions":
        if "none" in transcript_lower or "no" in transcript_lower or "healthy" in transcript_lower:
            extracted["medical_conditions"] = 0
        else:
            # Look for common conditions or numbers
            if "diabetes" in transcript_lower or "blood pressure" in transcript_lower or "asthma" in transcript_lower or "thyroid" in transcript_lower:
                extracted["medical_conditions"] = 1
            nums = re.findall(r'(\d+)', transcript_lower)
            if nums:
                valid_nums = [int(n) for n in nums if 0 <= int(n) <= 10]
                if valid_nums:
                    extracted["medical_conditions"] = valid_nums[0]

    return {
        "extracted_features": extracted,
        "transcript": transcript,
        "context": context,
        "sentiment_breakdown": vader_scores,
        "keyword_flags": keyword_analysis["high_risk_terms_found"]
    }
