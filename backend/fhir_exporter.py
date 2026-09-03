"""
FHIR / HL7 R4 Interoperability Exporter Module.
Generates standard FHIR R4 JSON bundles for Epic / Cerner hospital EHR systems integration.
"""

import json
import time
from typing import Dict, Any

def generate_fhir_r4_bundle(patient_id: str, patient_name: str, age: int, gender: str, risk_scores: Dict[str, float], patient_features: Dict[str, Any], doctor_cert: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a standard FHIR R4 Bundle containing Patient, Observations, DiagnosticReport, and Practitioner resources.
    """
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    
    # 1. FHIR Patient Resource
    patient_resource = {
        "resourceType": "Patient",
        "id": patient_id.replace("MRN-", "pat-"),
        "identifier": [
            {
                "system": "urn:oid:2.16.840.1.113883.2.4.6.3",
                "value": patient_id
            }
        ],
        "active": True,
        "name": [
            {
                "use": "official",
                "text": patient_name,
                "family": patient_name.split()[-1] if " " in patient_name else patient_name,
                "given": [patient_name.split()[0]]
            }
        ],
        "gender": gender.lower() if gender.lower() in ["male", "female", "other"] else "unknown"
    }

    # 2. FHIR Observations (PHQ-9, Risk Scores)
    observations = [
        {
            "resourceType": "Observation",
            "id": f"obs-phq9-{patient_id}",
            "status": "final",
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "44261-6",
                        "display": "Patient Health Questionnaire 9 item (PHQ-9) total score"
                    }
                ]
            },
            "subject": {"reference": f"Patient/{patient_resource['id']}"},
            "effectiveDateTime": timestamp,
            "valueQuantity": {
                "value": float(patient_features.get("phq9_score", 0)),
                "unit": "{score}",
                "system": "http://unitsofmeasure.org"
            }
        },
        {
            "resourceType": "Observation",
            "id": f"obs-dep-risk-{patient_id}",
            "status": "final",
            "code": {
                "coding": [
                    {
                        "system": "http://snomed.info/sct",
                        "code": "35489007",
                        "display": "Depressive disorder risk score"
                    }
                ]
            },
            "subject": {"reference": f"Patient/{patient_resource['id']}"},
            "effectiveDateTime": timestamp,
            "valueQuantity": {
                "value": float(risk_scores.get("depression", 0)),
                "unit": "%",
                "system": "http://unitsofmeasure.org"
            }
        }
    ]

    # 3. FHIR Practitioner Resource
    practitioner_resource = {
        "resourceType": "Practitioner",
        "id": "prac-vedant-sharma",
        "identifier": [
            {
                "system": "urn:oid:2.16.840.1.113883.4.6",
                "value": doctor_cert.get("license_number", "MP-88421-IN")
            }
        ],
        "name": [
            {
                "text": doctor_cert.get("doctor_name", "Dr. Vedant Sharma, MD"),
                "prefix": ["Dr."]
            }
        ],
        "qualification": [
            {
                "code": {
                    "text": doctor_cert.get("qualification", "MD Psychiatry")
                }
            }
        ]
    }

    # 4. FHIR DiagnosticReport Resource
    diagnostic_report = {
        "resourceType": "DiagnosticReport",
        "id": f"diagrep-{patient_id}",
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                        "code": "PSY",
                        "display": "Psychiatry Clinical Decision Support"
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "80826-1",
                    "display": "Psychiatric clinical AI decision support evaluation"
                }
            ],
            "text": "Multi-Modal AI Clinical Decision Support Assessment"
        },
        "subject": {"reference": f"Patient/{patient_resource['id']}"},
        "issued": timestamp,
        "performer": [{"reference": f"Practitioner/{practitioner_resource['id']}"}],
        "result": [{"reference": f"Observation/{obs['id']}"} for obs in observations],
        "conclusion": f"Depression Risk: {risk_scores.get('depression', 0)}%, Anxiety Risk: {risk_scores.get('anxiety', 0)}%, Suicide Alert: {risk_scores.get('suicide', 0)}%. Digital Verification Hash: {doctor_cert.get('digital_signature_hash', '')}"
    }

    # Assemble FHIR Bundle
    bundle = {
        "resourceType": "Bundle",
        "id": f"bundle-fhir-{patient_id}",
        "meta": {"lastUpdated": timestamp},
        "type": "document",
        "entry": [
            {"resource": patient_resource},
            {"resource": practitioner_resource},
            {"resource": diagnostic_report}
        ] + [{"resource": obs} for obs in observations]
    }

    return bundle
