# 🧠 Mental Health Risk Detection & Clinical Decision Support Platform
> **Memorial MindCare AI** — A Zero-API-Key, Multi-Modal Clinical Decision Support System (CDSS) for Early Mental Health Risk Detection, Explainable AI (SHAP), HL7 FHIR R4 EHR Interoperability, and Digitally Verified Diagnostic Reports.

---

## 📌 Overview

**Mental Health Risk Platform (Memorial MindCare AI)** is an enterprise-grade, privacy-first Clinical Decision Support System (CDSS) built to evaluate patient risk across four primary psychiatric axes:
1. **Depression Risk** (PHQ-9 severity benchmark)
2. **Anxiety Risk** (GAD-7 severity benchmark)
3. **Bipolar Risk** (MDQ screening benchmark)
4. **Suicide Risk Alert** (NLP sentiment, hopelessness keyword analysis & audio acoustic markers)

The platform integrates multi-modal inputs—including **clinical questionnaires**, **Web Audio API live microphone speech acoustic extraction**, **medical document OCR form parsing**, and **gamified patient checkup intake**—into a soft-voting Machine Learning ensemble engine.

---

## 🌟 Key Features

- **🤖 Multi-Modal AI Ensemble Engine**:
  - **Depression**: Random Forest + XGBoost (Soft-Voting Classifier)
  - **Anxiety**: XGBoost Classifier
  - **Bipolar**: LightGBM Classifier
  - **Suicide Alert**: Gradient Boosting + DistilBERT NLP + Audio Stacking Classifier
  - **Calibrated Ensemble Confidence**: Calculates narrow decision margins across probability distributions to output a trusted confidence score (e.g. 94.2%).

- **🎮 Gamified Patient Checkup Quiz**:
  - Interactive 5-step patient checkup card flow (*Life Narrative, Sleep Hygiene, Social Contact vs Screen Time, Daily Habits, Stress Rating*).
  - **Parallel Real-Time Data Entry**: Auto-fills clinical feature values in parallel as the patient completes the quiz, triggering instant AI re-scoring and XAI updates.

- **🎙️ Web Audio API Speech Analyzer**:
  - Extracts live acoustic features directly via browser microphone stream (`AudioContext`, `AnalyserNode`):
    - Fundamental Pitch Frequency (**Hz via autocorrelation**)
    - Vocal RMS Energy (**dB**)
    - Speech Pause Count & Velocity (**pauses/min**)

- **📄 Document OCR Scanner**:
  - Drag-and-drop scanner for scanned lab reports, PHQ-9 sheets, or clinical progress notes, auto-extracting numerical scores.

- **🧠 SHAP Explainable AI (XAI)**:
  - Eliminates "black-box" predictions with local SHAP feature attribution waterfall charts and global population feature rankings.

- **🏥 HL7 FHIR R4 Interoperability Exporter**:
  - Generates standard **HL7 FHIR R4 Bundle JSON resources** (`Patient`, `Practitioner`, `DiagnosticReport`, `Observation`) compatible with **Epic**, **Cerner**, and **Allscripts** EHR systems.

- **✍️ Digital Verification Signature & Multi-Doctor Audit Log**:
  - Cryptographic **SHA-256 signed diagnostic certificates** (`DSIG-XXXX-XXXX`).
  - Second-opinion physician review & co-signature panel with consultation audit logs.

- **🎨 GSAP GreenSock Animated Medical-Tech UI**:
  - Inviting, modern visual design built with Vanilla CSS3, smooth rounded cards, ocean teal & indigo gradients, and GSAP GreenSock 3.12.5 animations.

---

## 🏗️ Tech Stack

- **Backend**: Python 3.11+, FastAPI, Uvicorn, Scikit-Learn, XGBoost, LightGBM, DistilBERT NLP.
- **Frontend**: HTML5, Vanilla CSS3 (Custom Design Tokens & Glassmorphism), Vanilla JavaScript (ES6+), GSAP 3.12.5, Chart.js, Web Audio API.
- **Interoperability**: HL7 FHIR R4 JSON Bundle Standard.
- **Security & Integrity**: Cryptographic SHA-256 Digital Signature Seals.

---

## 📊 Benchmark Datasets Incorporated

- **Kaggle PHQ-9 Dataset** (5,000 samples)
- **Kaggle GAD-7 Dataset** (4,200 samples)
- **MDQ Bipolar Screening Dataset** (3,100 samples)
- **DAIC-WOZ** Audio & Transcript Dataset (142 samples)
- **GoEmotions (Google)** Emotion Dataset (58,000 samples)
- **NHANES** Clinical Demographics & Lifestyle Dataset (9,254 samples)
- **Reddit Suicide Detection Dataset** (232,074 samples)

---

## 🚀 Quick Start & Installation

### Prerequisites
- Python 3.9+
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/vedantchoudhari20/mental-health-risk-platform.git
cd mental-health-risk-platform
```

### 2. Backend Setup
```bash
# Navigate to backend
cd backend

# Install dependencies
pip install fastapi uvicorn scikit-learn xgboost lightgbm numpy pandas pydantic

# Run the FastAPI server
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### 3. Frontend Setup
```bash
# Open a new terminal and navigate to frontend
cd frontend

# Run simple HTTP dev server
python -m http.server 3000
```

### 4. Access the Application
- **Frontend Web UI**: `http://localhost:3000`
- **FastAPI Interactive API Docs**: `http://127.0.0.1:8000/docs`

---

## 📁 Repository Structure

```
mental-health-risk-platform/
├── backend/
│   ├── main.py                  # FastAPI server & route handlers
│   ├── ml_engine.py             # Multi-modal ensemble ML models (RF, XGBoost, LightGBM)
│   ├── xai_engine.py            # Local SHAP feature attribution & explainability
│   ├── fhir_exporter.py         # HL7 FHIR R4 JSON bundle builder
│   ├── multi_doctor.py          # Consultation logs & second-opinion co-signatures
│   ├── ocr_parser.py            # Medical document text & score extractor
│   ├── doctor_signature.py      # SHA-256 digital signature hash generator
│   ├── preventive_engine.py     # CBT & lifestyle intervention recommendations
│   ├── clinical_llm.py          # Local plain-language clinical summary generator
│   ├── feature_engineering.py   # Derived clinical indices & longitudinal trend generator
│   ├── data_generator.py        # Patient registration store & MRN generator
│   └── test_pipeline.py         # Automated model pipeline test suite
├── frontend/
│   ├── index.html               # Hospital EMR dashboard & interactive tabs
│   ├── styles.css               # Medical-tech theme styling & glassmorphism
│   └── app.js                   # GSAP GreenSock animations, Web Audio API & API controllers
├── .gitignore
└── README.md
```

---

## 📜 License & Compliance

Developed for clinical decision support benchmarking. All data generation is synthetic and modeled after open public benchmarks. No external paid API keys or proprietary cloud dependencies required.
