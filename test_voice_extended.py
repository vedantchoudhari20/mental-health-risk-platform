import requests
import json

url = "http://127.0.0.1:8000/api/voice-intake"

test_cases = [
    {
        "context": "family",
        "transcript": "Yes, my mother had severe depression."
    },
    {
        "context": "episodes",
        "transcript": "I have never felt like this before."
    },
    {
        "context": "medications",
        "transcript": "I am currently taking two pills for blood pressure."
    },
    {
        "context": "conditions",
        "transcript": "I have diabetes and thyroid issues."
    }
]

print("=== LIVE CLINICAL NLP EXTENDED TEST ===\n")

for i, case in enumerate(test_cases):
    print(f"Test Case {i+1}: Context='{case['context']}'")
    print(f"Patient Transcript: \"{case['transcript']}\"")
    
    resp = requests.post(url, json=case)
    if resp.status_code == 200:
        data = resp.json()
        print("-> Extracted Features:", json.dumps(data["extracted_features"], indent=2))
        print("-" * 50)
    else:
        print("Error:", resp.text)
