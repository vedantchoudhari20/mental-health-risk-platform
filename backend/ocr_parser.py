"""
OCR & Medical Document Parser Module — with VADER NLP Sentiment Analysis.

Parses scanned medical documents, lab reports, questionnaire PDFs, or clinical text notes
to automatically extract clinical multi-modal parameters (PHQ-9, GAD-7, MDQ, sleep, stress, etc.).

Sentiment analysis uses VADER (Valence Aware Dictionary and sEntiment Reasoner),
a validated rule-based model for social media and clinical text sentiment analysis.

Reference: Hutto & Gilbert (2014). VADER: A Parsimonious Rule-based Model for
Sentiment Analysis of Social Media Text. ICWSM.
"""

import re
from typing import Dict, Any

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    _VADER_AVAILABLE = True
except ImportError:
    _vader = None
    _VADER_AVAILABLE = False


# ─────────────────────────────────────────────────────────
# Clinical Distress Keyword Lexicon (expanded)
# ─────────────────────────────────────────────────────────

# High-risk terms (weighted higher for keyword_intensity)
HIGH_RISK_TERMS = [
    "suicidal", "suicide", "kill myself", "end my life", "want to die",
    "better off dead", "no reason to live", "can't go on", "self-harm",
    "cutting", "overdose",
]

# Moderate distress terms
MODERATE_DISTRESS_TERMS = [
    "hopeless", "worthless", "helpless", "empty", "numb", "despair",
    "depressed", "devastated", "miserable", "trapped", "burden",
    "isolated", "alone", "abandoned", "rejected", "useless",
]

# Mild distress terms
MILD_DISTRESS_TERMS = [
    "sad", "tired", "exhausted", "anxious", "worried", "stressed",
    "overwhelmed", "frustrated", "irritable", "nervous", "crying",
    "sleepless", "insomnia", "fatigue", "drained", "struggle",
    "difficulty", "can't sleep", "can't focus", "can't concentrate",
]

# Positive / protective terms
POSITIVE_TERMS = [
    "happy", "energetic", "good", "calm", "improving", "stable",
    "optimistic", "hopeful", "grateful", "confident", "motivated",
    "peaceful", "relaxed", "content", "engaged", "active",
    "connected", "supported", "better",
]


def _analyze_sentiment_vader(text: str) -> Dict[str, float]:
    """
    Uses VADER for validated sentiment analysis with proper handling of:
    - Negation ("not happy" → negative)
    - Degree modifiers ("extremely sad" vs "slightly sad")
    - Punctuation emphasis ("I'm so tired!!!")
    - Capitalization ("TERRIBLE" vs "terrible")
    - Conjunctions ("I feel good but also very anxious")
    """
    if not _VADER_AVAILABLE or not text.strip():
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}

    scores = _vader.polarity_scores(text)
    return {
        "compound": scores["compound"],      # -1.0 to +1.0
        "pos": scores["pos"],                 # 0.0 to 1.0
        "neg": scores["neg"],                 # 0.0 to 1.0
        "neu": scores["neu"],                 # 0.0 to 1.0
    }


def _compute_clinical_keyword_analysis(text: str) -> Dict[str, Any]:
    """
    Performs clinical-focused keyword analysis with severity weighting.
    """
    text_lower = text.lower()

    high_matches = [t for t in HIGH_RISK_TERMS if t in text_lower]
    moderate_matches = [t for t in MODERATE_DISTRESS_TERMS if t in text_lower]
    mild_matches = [t for t in MILD_DISTRESS_TERMS if t in text_lower]
    positive_matches = [t for t in POSITIVE_TERMS if t in text_lower]

    # Weighted keyword intensity (0-100)
    # High-risk terms contribute 30 each, moderate 15 each, mild 8 each
    # Positive terms reduce by 5 each
    raw_intensity = (
        len(high_matches) * 30 +
        len(moderate_matches) * 15 +
        len(mild_matches) * 8 -
        len(positive_matches) * 5
    )
    keyword_intensity = float(max(0, min(100, raw_intensity)))

    # Sadness probability estimate
    sadness_indicators = len([t for t in moderate_matches + mild_matches
                              if t in ("sad", "depressed", "hopeless", "empty", "miserable",
                                       "crying", "devastated", "numb", "despair")])
    sadness_prob = min(1.0, sadness_indicators * 0.2)

    # Suicide risk flag
    suicide_flag = len(high_matches) > 0

    return {
        "keyword_intensity": keyword_intensity,
        "sadness_prob": sadness_prob,
        "suicide_risk_flag": suicide_flag,
        "high_risk_terms_found": high_matches,
        "moderate_distress_terms_found": moderate_matches,
        "mild_distress_terms_found": mild_matches,
        "positive_terms_found": positive_matches,
        "total_distress_terms": len(high_matches) + len(moderate_matches) + len(mild_matches),
        "total_positive_terms": len(positive_matches),
    }


def parse_medical_document_text(text_content: str) -> Dict[str, Any]:
    """
    Scans raw text extracted from medical lab reports, questionnaire sheets, or clinical notes.
    Uses clinical pattern matching to auto-extract assessment values.
    Uses VADER for validated sentiment analysis.
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

    # 7. VADER Sentiment Analysis
    vader_scores = _analyze_sentiment_vader(text_content)
    extracted["sentiment_score"] = round(vader_scores["compound"], 4)

    # 8. Clinical Keyword Analysis
    keyword_analysis = _compute_clinical_keyword_analysis(text_content)
    extracted["sadness_prob"] = keyword_analysis["sadness_prob"]
    extracted["keyword_intensity"] = keyword_analysis["keyword_intensity"]
    extracted["hopelessness_prob"] = min(1.0, keyword_analysis["keyword_intensity"] / 100 * 0.8)

    return {
        "extracted_features": extracted,
        "fields_found": list(extracted.keys()),
        "raw_text_length": len(text_content),
        "vader_sentiment": vader_scores,
        "keyword_analysis": keyword_analysis,
        "nlp_engine": "VADER (Hutto & Gilbert 2014)" if _VADER_AVAILABLE else "Fallback keyword matching",
        "status": "PARSED_SUCCESSFULLY",
    }
