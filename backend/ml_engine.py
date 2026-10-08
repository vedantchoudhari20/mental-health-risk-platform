"""
ML Engine Module - Multi-Modal Machine Learning Ensemble & Benchmark Evaluator.
Trains Random Forest, XGBoost, LightGBM, and Stacking/Voting Classifiers for Mental Health Risk Detection.

IMPORTANT CHANGES (v2.0 — Clinical Integrity Rewrite):
  - Uses proper 80/20 train-test split for honest performance metrics.
  - Returns cross-validated accuracy metrics (not training data accuracy).
  - Calculates confidence scores using Platt scaling (CalibratedClassifierCV),
    providing mathematically grounded probability bounds.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, cohen_kappa_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb

from data_generator import generate_synthetic_cohort, FEATURE_COLUMNS
from feature_engineering import compute_derived_features

class MentalHealthMLEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.metrics = {}
        self.is_trained = False
        self.feature_names = FEATURE_COLUMNS

    def train_pipeline(self, n_samples: int = 3000) -> Dict[str, Any]:
        """
        Trains multi-modal ensemble classifiers for Depression, Anxiety, Bipolar, and Suicide Risk.
        Calculates honest performance benchmarks using a held-out test set.
        """
        X_raw, targets, _ = generate_synthetic_cohort(n_samples=n_samples)
        
        # Proper Train/Test Split (80/20)
        indices = np.arange(n_samples)
        X_train_raw, X_test_raw, idx_train, idx_test = train_test_split(
            X_raw, indices, test_size=0.20, random_state=42
        )
        
        # Fit scaler on train only
        X_train_scaled = self.scaler.fit_transform(X_train_raw)
        X_test_scaled = self.scaler.transform(X_test_raw)

        # 1. Depression Model (Random Forest + XGBoost Ensemble, Calibrated)
        rf_dep = RandomForestClassifier(n_estimators=120, max_depth=10, random_state=42)
        xgb_dep = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.08, eval_metric='logloss', random_state=42)
        dep_ensemble = VotingClassifier(estimators=[('rf', rf_dep), ('xgb', xgb_dep)], voting='soft')
        calibrated_dep = CalibratedClassifierCV(estimator=dep_ensemble, method='sigmoid', cv=5)
        calibrated_dep.fit(X_train_scaled, targets['depression'][idx_train])
        self.models['depression'] = calibrated_dep

        # 2. Anxiety Model (XGBoost Classifier, Calibrated)
        xgb_anx = xgb.XGBClassifier(n_estimators=130, max_depth=6, learning_rate=0.07, eval_metric='logloss', random_state=42)
        calibrated_anx = CalibratedClassifierCV(estimator=xgb_anx, method='sigmoid', cv=5)
        calibrated_anx.fit(X_train_scaled, targets['anxiety'][idx_train])
        self.models['anxiety'] = calibrated_anx

        # 3. Bipolar Risk Model (LightGBM Classifier, Calibrated)
        lgb_bip = lgb.LGBMClassifier(n_estimators=110, max_depth=5, learning_rate=0.06, random_state=42, verbose=-1)
        calibrated_bip = CalibratedClassifierCV(estimator=lgb_bip, method='sigmoid', cv=5)
        calibrated_bip.fit(X_train_scaled, targets['bipolar'][idx_train])
        self.models['bipolar'] = calibrated_bip

        # 4. Suicide Risk Model (Gradient Boosting, Calibrated)
        gb_sui = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42)
        calibrated_sui = CalibratedClassifierCV(estimator=gb_sui, method='sigmoid', cv=5)
        calibrated_sui.fit(X_train_scaled, targets['suicide'][idx_train])
        self.models['suicide'] = calibrated_sui

        # Calculate Validation Benchmarks ON HOLDOUT TEST SET ONLY
        test_targets = {
            'depression': targets['depression'][idx_test],
            'anxiety': targets['anxiety'][idx_test],
            'bipolar': targets['bipolar'][idx_test],
            'suicide': targets['suicide'][idx_test],
        }
        self.metrics = self._evaluate_performance(X_test_scaled, test_targets)
        self.is_trained = True
        return self.metrics

    def _evaluate_performance(self, X_scaled: np.ndarray, targets: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Evaluates model metrics on held-out test data.
        """
        eval_summary = {}

        for key in ['depression', 'anxiety', 'bipolar', 'suicide']:
            model = self.models[key]
            y_true = targets[key]
            y_pred = model.predict(X_scaled)
            y_prob = model.predict_proba(X_scaled)[:, 1]

            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()

            sens = recall_score(y_true, y_pred)
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_prob)
            kappa = cohen_kappa_score(y_true, y_pred)

            eval_summary[key] = {
                "accuracy": round(float(acc), 4),
                "sensitivity": round(float(sens), 4),
                "specificity": round(float(spec), 4),
                "roc_auc": round(float(auc), 4),
                "f1_score": round(float(f1), 4),
                "kappa_score": round(float(kappa), 4)
            }

        eval_summary["final_ensemble"] = {
            "accuracy": round(float(np.mean([m['accuracy'] for m in eval_summary.values()])), 4),
            "sensitivity": round(float(np.mean([m['sensitivity'] for m in eval_summary.values()])), 4),
            "specificity": round(float(np.mean([m['specificity'] for m in eval_summary.values()])), 4),
            "roc_auc": round(float(np.mean([m['roc_auc'] for m in eval_summary.values()])), 4),
        }
        return eval_summary

    def predict_patient_risk(self, patient_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs inference across 4 risk axes. Returns calibrated probabilities.
        """
        if not self.is_trained:
            self.train_pipeline()

        # Build feature vector
        x_vec = np.zeros((1, len(self.feature_names)))
        for i, col in enumerate(self.feature_names):
            x_vec[0, i] = float(patient_features.get(col, 0.0))

        x_scaled = self.scaler.transform(x_vec)

        # 1. Depression Prediction
        dep_prob = float(self.models['depression'].predict_proba(x_scaled)[0, 1])

        # 2. Anxiety Prediction
        anx_prob = float(self.models['anxiety'].predict_proba(x_scaled)[0, 1])

        # 3. Bipolar Prediction
        bip_prob = float(self.models['bipolar'].predict_proba(x_scaled)[0, 1])

        # 4. Suicide Risk Prediction (incorporates NLP flags explicitly)
        sui_model_prob = float(self.models['suicide'].predict_proba(x_scaled)[0, 1])
        hopeless_prob = float(patient_features.get("hopelessness_prob", 0.0))
        keyword_int = float(patient_features.get("keyword_intensity", 0.0)) / 100.0
        
        # Clinical override for suicide risk based on explicit high-risk terms
        if patient_features.get("suicide_risk_flag", False):
            sui_prob = 0.95
        else:
            sui_prob = float(np.clip(0.60 * sui_model_prob + 0.25 * hopeless_prob + 0.15 * keyword_int, 0.0, 1.0))

        # Risk Category Mapper
        def get_category(prob: float) -> str:
            if prob < 0.35:
                return "LOW"
            elif prob < 0.70:
                return "MODERATE"
            else:
                return "HIGH"

        derived = compute_derived_features(patient_features)

        risk_scores = {
            "depression": round(dep_prob * 100, 1),
            "anxiety": round(anx_prob * 100, 1),
            "bipolar": round(bip_prob * 100, 1),
            "suicide": round(sui_prob * 100, 1)
        }

        risk_categories = {
            "depression": get_category(dep_prob),
            "anxiety": get_category(anx_prob),
            "bipolar": get_category(bip_prob),
            "suicide": get_category(sui_prob)
        }

        # Honest Confidence Score:
        # Distance from 0.5 (uncertainty boundary). Calibrated models output true probabilities.
        # So a probability of 0.50 has 0% confidence, 0.99 has 98% confidence.
        distances = [abs(p - 0.5) * 2 for p in [dep_prob, anx_prob, bip_prob, sui_prob]]
        mean_certainty = float(np.mean(distances))
        calibrated_confidence = round(mean_certainty * 100, 1)

        has_high = any(cat == "HIGH" for cat in risk_categories.values())
        has_suicide_alert = risk_scores["suicide"] >= 50.0

        algorithm_engagement = {
            "depression_algorithms": ["Calibrated RandomForest + XGBoost"],
            "anxiety_algorithms": ["Calibrated XGBoost"],
            "bipolar_algorithms": ["Calibrated LightGBM"],
            "suicide_algorithms": ["Calibrated GradientBoosting + NLP Fusion"],
            "derived_feature_engine": ["Sleep Risk Index", "Social Isolation Index", "Mood Variability Index", "Depression Composite Index"]
        }

        return {
            "risk_scores": risk_scores,
            "risk_categories": risk_categories,
            "derived_indices": derived,
            "alert_triggered": has_high or has_suicide_alert,
            "alert_level": "CRITICAL" if has_suicide_alert or risk_scores["depression"] >= 80.0 else ("WARNING" if has_high else "STABLE"),
            "confidence_score": calibrated_confidence,
            "algorithm_engagement": algorithm_engagement
        }

# Global singleton engine instance
ml_engine = MentalHealthMLEngine()
