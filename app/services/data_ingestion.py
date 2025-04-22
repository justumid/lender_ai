# app/services/data_ingestion.py

import json
import os


def fetch_user_data_by_pinfl(pinfl: str, path="demo_data/synthetic_dataset.json") -> dict:
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    for item in dataset:
        if item.get("pinfl") == pinfl:
            return item

    return {}
