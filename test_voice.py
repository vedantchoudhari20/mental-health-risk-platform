import requests
import json

url = "http://127.0.0.1:8000/api/voice-intake"

test_cases = [
    {
        "context": "sleep",
        "transcript": "I have been struggling a lot, I only sleep about 3.5 hours and wake up exhausted."
    },
    {
        "context": "mood",
        "transcript": "Honestly I feel completely hopeless. Nothing makes me happy anymore, just empty and sad."
    },
    {
        "context": "activity",
        "transcript": "I try to walk my dog for 45 minutes every morning, but that's it."
    },
    {
        "context": "stress",
        "transcript": "My stress is through the roof, I'd say it's a solid 9 out of 10 right now."
    },
    {
        "context": "mood",
        "transcript": "I am feeling fantastic! Life is going really well and I'm very energetic and calm."
    }
]

print("=== LIVE CLINICAL NLP PIPELINE TEST ===\n")

for i, case in enumerate(test_cases):
    print(f"Test Case {i+1}: Context='{case['context']}'")
    print(f"Patient Transcript: \"{case['transcript']}\"")
    
    resp = requests.post(url, json=case)
    if resp.status_code == 200:
        data = resp.json()
        print("-> Extracted Features:", json.dumps(data["extracted_features"], indent=2))
        print("-> Sentiment Breakdown:", json.dumps(data["sentiment_breakdown"], indent=2))
        print("-" * 50)
    else:
        print("Error:", resp.text)
