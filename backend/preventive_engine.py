"""
Preventive Measures Engine - Clinical Rule Engine & Evidence-Based Interventions.
Generates tailored preventive care recommendations, CBT exercises, lifestyle interventions, and follow-up schedules.
"""

from typing import Dict, Any, List

def generate_preventive_care_plan(patient_features: Dict[str, Any], risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates personalized clinical preventive actions based on risk profile and top contributing factors.
    """
    risk_scores = risk_assessment.get("risk_scores", {})
    categories = risk_assessment.get("risk_categories", {})
    derived = risk_assessment.get("derived_indices", {})

    dep_risk = risk_scores.get("depression", 0.0)
    anx_risk = risk_scores.get("anxiety", 0.0)
    bip_risk = risk_scores.get("bipolar", 0.0)
    sui_risk = risk_scores.get("suicide", 0.0)

    sleep_hrs = float(patient_features.get("sleep_hours", 7.0))
    activity_min = float(patient_features.get("physical_activity_min", 30))
    phq9 = float(patient_features.get("phq9_score", 0))

    actions = []
    cbt_interventions = []
    lifestyle = []
    clinical_referrals = []

    # 1. Sleep Hygiene Rules
    if sleep_hrs < 6.5 or sleep_hrs > 9.5 or derived.get("sleep_risk_index", 0) > 40:
        actions.append({
            "category": "Lifestyle",
            "action": "Sleep Hygiene Optimization (Target: 7-8 hrs)",
            "priority": "HIGH",
            "detail": "Maintain consistent sleep-wake cycle, restrict blue light 1 hour prior to sleep, avoid caffeine after 2 PM."
        })
        lifestyle.append("Sleep Hygiene (7-8 hrs nightly baseline)")
    
    # 2. Physical Exercise Rules
    if activity_min < 30:
        actions.append({
            "category": "Behavioral",
            "action": "Daily Physical Movement (30 min Daily Walk)",
            "priority": "HIGH",
            "detail": "Engage in 30 minutes of moderate aerobic activity (brisk walking, light cycling) to stimulate neurotrophic factors (BDNF)."
        })
        lifestyle.append("30 min Daily Brisk Walk")

    # 3. CBT & Psychological Interventions
    if dep_risk >= 35.0 or anx_risk >= 35.0 or phq9 >= 10:
        actions.append({
            "category": "Therapeutic",
            "action": "Cognitive Behavioral Therapy (CBT) & Relaxation",
            "priority": "HIGH",
            "detail": "Structured CBT thought record journaling for cognitive reframing + 4-7-8 diaphragmatic breathing exercises 2x daily."
        })
        cbt_interventions.append("CBT Thought Restructuring & Relaxation Exercises")

    # 4. Clinical / Psychiatric Consultation
    if dep_risk >= 70.0 or anx_risk >= 70.0 or bip_risk >= 50.0:
        actions.append({
            "category": "Clinical",
            "action": "Psychiatric Consultation & Medication Review",
            "priority": "CRITICAL",
            "detail": "Referral to attending psychiatrist for comprehensive diagnostic review and potential pharmacotherapy evaluation."
        })
        clinical_referrals.append("Psychiatric Consultation within 48-72 hours")
    elif dep_risk >= 50.0 or anx_risk >= 50.0:
        clinical_referrals.append("Follow-up Consultation in 7 Days")

    # 5. Urgent Crisis Safety Protocol
    if sui_risk >= 50.0 or categories.get("suicide") == "HIGH":
        actions.insert(0, {
            "category": "Safety Alert",
            "action": "Immediate Crisis Assessment & Safety Plan Dispatch",
            "priority": "URGENT",
            "detail": "Activate hospital crisis protocol, conduct immediate safety assessment (Columbia Scale), notify clinical care coordinator."
        })
        clinical_referrals.insert(0, "Immediate Hospital Crisis Team Review")
        followup_timeline = "Immediate (< 24 Hours)"
    elif dep_risk >= 70.0:
        followup_timeline = "Follow-up in 3 Days"
    elif dep_risk >= 40.0 or anx_risk >= 40.0:
        followup_timeline = "Follow-up in 7 Days"
    else:
        followup_timeline = "Routine Follow-up in 14 Days"

    # Default fallback actions to ensure complete care plan (matching diagram example)
    if len(actions) < 4:
        actions.append({
            "category": "Social",
            "action": "Social Re-engagement & Community Support",
            "priority": "MEDIUM",
            "detail": "Schedule structured weekly social interactions with family/peer support groups to mitigate isolation."
        })
        lifestyle.append("Social Re-engagement")

    recommended_summary_list = [
        "Sleep Hygiene (7-8 hrs)",
        "30 min Daily Walk",
        "CBT / Relaxation Exercises",
        "Psychiatric Consultation" if dep_risk >= 60 else "Mindfulness Meditation",
        f"Clinical Follow-up in {followup_timeline.split()[-2]} Days" if "Days" in followup_timeline else "Follow-up in 7 Days"
    ]

    return {
        "recommended_actions": recommended_summary_list,
        "detailed_action_plan": actions,
        "lifestyle_recommendations": lifestyle,
        "cbt_modules": cbt_interventions,
        "clinical_referrals": clinical_referrals,
        "followup_timeline": followup_timeline,
        "hospital_alert": sui_risk >= 50.0 or dep_risk >= 80.0
    }
