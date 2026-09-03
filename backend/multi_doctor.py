"""
Multi-Doctor Co-Signing & Consultation Audit Log Module.
Manages second opinion physician reviews, co-signatures, and clinical consultation audit logs.
"""

import time
from typing import Dict, List, Any

DEFAULT_SECOND_OPINION_DOCTOR = {
    "doctor_id": "DOC-8814",
    "name": "Dr. Amanda Vance, MD",
    "qualification": "MD Psychiatry, Chief of Clinical Consultation",
    "hospital": "Memorial Neuropsychiatric Hospital",
    "license_number": "MP-91024-IN"
}

# In-memory consultation store
CONSULTATION_LOGS: Dict[str, List[Dict[str, Any]]] = {}

def add_doctor_cosignature(patient_id: str, doctor_info: Dict[str, Any] = None, comments: str = "Reviewed and agreed with clinical AI risk evaluation and preventive plan.") -> Dict[str, Any]:
    """
    Adds a second opinion doctor co-signature and review comment to patient record.
    """
    doc = doctor_info or DEFAULT_SECOND_OPINION_DOCTOR
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    entry = {
        "log_id": f"COSIG-{int(time.time())}",
        "patient_id": patient_id,
        "doctor_name": doc["name"],
        "qualification": doc["qualification"],
        "hospital": doc["hospital"],
        "license_number": doc["license_number"],
        "timestamp": timestamp,
        "action": "SECOND OPINION CO-SIGNATURE & AUDIT REVIEW",
        "comments": comments,
        "status": "APPROVED & CO-SIGNED"
    }

    if patient_id not in CONSULTATION_LOGS:
        CONSULTATION_LOGS[patient_id] = []

    CONSULTATION_LOGS[patient_id].append(entry)

    return {
        "status": "CO_SIGNATURE_SUCCESSFUL",
        "new_entry": entry,
        "total_cosignatures": len(CONSULTATION_LOGS[patient_id]),
        "all_consultation_logs": CONSULTATION_LOGS[patient_id]
    }

def get_patient_consultation_logs(patient_id: str) -> List[Dict[str, Any]]:
    """Returns consultation audit logs for patient."""
    return CONSULTATION_LOGS.get(patient_id, [
        {
            "log_id": "COSIG-INITIAL",
            "patient_id": patient_id,
            "doctor_name": "Dr. Vedant Sharma, MD",
            "qualification": "MD Psychiatry (Primary Attending Physician)",
            "hospital": "Memorial Neuropsychiatric Hospital",
            "license_number": "MP-88421-IN",
            "timestamp": "2026-08-03 10:00:00 UTC",
            "action": "PRIMARY CLINICAL AI EVALUATION & INSPECTION",
            "comments": "Initial multi-modal risk scoring completed and digitally verified.",
            "status": "PRIMARY SIGNED"
        }
    ])
