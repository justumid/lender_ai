import torch
import json
from app.services.data_ingestion import fetch_user_data_by_pinfl
from app.services.explanation import explain_attention

def explain_from_db(pinfl: str):
    applicant = fetch_user_data_by_pinfl(pinfl)
    if not applicant:
        print(f"[X] No applicant found for PINFL: {pinfl}")
        return

    attention_scores = explain_attention(applicant)
    print(json.dumps(attention_scores, indent=2))

if __name__ == "__main__":
    pinfl_to_debug = "20101862710039"  # 🔧 Replace with test PINFL
    explain_from_db(pinfl_to_debug)
