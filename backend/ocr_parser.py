"""
OCR & Medical Document Parser Module.
Parses scanned medical documents, lab reports, questionnaire PDFs, or clinical text notes
to automatically extract clinical multi-modal parameters (PHQ-9, GAD-7, MDQ, sleep, stress, etc.).
"""

import re
from typing import Dict, Any

def parse_medical_document_text(text_content: str) -> Dict[str, Any]:
    """
    Scans raw text extracted from medical lab reports, questionnaire sheets, or clinical notes.
    Uses clinical pattern matching to auto-extract assessment values.
    """
    text_lower = text_content.lower()
    extracted = {}

    # 1. PHQ-9 Score (e.g. "PHQ-9: 22" or "PHQ9 score = 18")
    phq_match = re.search(r'(?:phq-?9|depression score)\D*(\d{1,2})', text_lower)
    if phq_match:
        val = int(phq_match.group(1))
        if 0 <= val <= 27:
            extracted["phq9_score"] = float(val)

    # 2. GAD-7 Score (e.g. "GAD-7: 14" or "anxiety score: 12")
    gad_match = re.search(r'(?:gad-?7|anxiety score)\D*(\d{1,2})', text_lower)
    if gad_match:
        val = int(gad_match.group(1))
        if 0 <= val <= 21:
            extracted["gad7_score"] = float(val)

    # 3. MDQ Score (e.g. "MDQ: 3" or "bipolar screening: 8")
    mdq_match = re.search(r'(?:mdq|bipolar score)\D*(\d{1,2})', text_lower)
    if mdq_match:
        val = int(mdq_match.group(1))
        if 0 <= val <= 13:
            extracted["mdq_score"] = float(val)

    # 4. Sleep Hours (e.g. "sleep: 4.5 hrs" or "sleeping 5 hours")
    sleep_match = re.search(r'(?:sleep|sleeping|sleep duration)\D*(\d+(?:\.\d+)?)', text_lower)
    if sleep_match:
        val = float(sleep_match.group(1))
        if 2.0 <= val <= 16.0:
            extracted["sleep_hours"] = val

    # 5. Stress Level (e.g. "stress: 8/10" or "stress level = 7")
    stress_match = re.search(r'(?:stress|stress level)\D*(\d{1,2})', text_lower)
    if stress_match:
        val = int(stress_match.group(1))
        if 1 <= val <= 10:
            extracted["stress_level"] = val

    # 6. Physical Activity (e.g. "walk 15 min" or "activity: 30 mins")
    act_match = re.search(r'(?:activity|exercise|walk)\D*(\d{1,3})', text_lower)
    if act_match:
        val = int(act_match.group(1))
        if 0 <= val <= 300:
            extracted["physical_activity_min"] = float(val)

    # 7. Sentiment & Keyword Intensity Analysis from Text Notes
    negative_terms = ["exhausted", "hopeless", "sad", "depressed", "worthless", "suicidal", "crying", "isolated", "tired", "anxious"]
    positive_terms = ["happy", "energetic", "good", "calm", "improving", "stable", "optimistic"]

    neg_count = sum(1 for term in negative_terms if term in text_lower)
    pos_count = sum(1 for term in positive_terms if term in text_lower)

    if neg_count > 0 or pos_count > 0:
        sent = float(np_clip_val((pos_count - neg_count) / max(1, pos_count + neg_count), -1.0, 1.0))
        extracted["sentiment_score"] = sent
        extracted["sadness_prob"] = float(np_clip_val(neg_count * 0.2, 0.0, 1.0))
        extracted["keyword_intensity"] = float(min(100, neg_count * 25))

    return {
        "extracted_features": extracted,
        "fields_found": list(extracted.keys()),
        "raw_text_length": len(text_content),
        "status": "PARSED_SUCCESSFULLY"
    }

def np_clip_val(v, min_v, max_v):
    return max(min_v, min(max_v, v))
