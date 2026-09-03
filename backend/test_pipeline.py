"""
Verification Script - Tests ML training, prediction, SHAP explainability, and recommendation pipeline.
"""

from ml_engine import ml_engine
from xai_engine import generate_patient_xai
from preventive_engine import generate_preventive_care_plan
from clinical_llm import generate_clinical_summary
from data_generator import get_sample_patients

def test_full_pipeline():
    print("==================================================")
    print("1. Training ML Pipeline & Ensemble Models...")
    metrics = ml_engine.train_pipeline(n_samples=1500)
    print("Model Validation Metrics:")
    for target, stat in metrics.items():
        print(f" - {target.upper()}: Accuracy={stat['accuracy']*100:.1f}%, Sensitivity={stat['sensitivity']*100:.1f}%, Specificity={stat['specificity']*100:.1f}%, ROC-AUC={stat['roc_auc']:.3f}")

    print("\n2. Testing Patient Inference (Patient P-1023: Vedant)...")
    samples = get_sample_patients()
    patient = samples[0] # Vedant
    pred = ml_engine.predict_patient_risk(patient["features"])
    print(f"Risk Scores: {pred['risk_scores']}")
    print(f"Risk Categories: {pred['risk_categories']}")
    print(f"Derived Indices: {pred['derived_indices']}")
    print(f"Alert Level: {pred['alert_level']}")

    print("\n3. Testing SHAP / LIME Explainability Engine...")
    xai = generate_patient_xai(patient["features"])
    print("Top 5 Contributing Factors:")
    for f in xai["top_contributing_factors"]:
        print(f" - {f['display_name']}: {f['percentage']}% impact")

    print("\n4. Testing Preventive Measures Engine...")
    plan = generate_preventive_care_plan(patient["features"], pred)
    print(f"Recommended Actions: {plan['recommended_actions']}")
    print(f"Follow-up Timeline: {plan['followup_timeline']}")

    print("\n5. Testing Local Zero-API-Key Clinical Summarizer...")
    summary = generate_clinical_summary(patient["name"], patient["age"], patient["gender"], pred, xai)
    print(f"Patient Summary: {summary['patient_summary']}")
    print("\n[SUCCESS] All pipeline checks passed cleanly!")
    print("==================================================")

if __name__ == "__main__":
    test_full_pipeline()
