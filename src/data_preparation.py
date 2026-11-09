import json
import pandas as pd
from pathlib import Path


def load_sft_data(file_path):
    """Load supervised fine-tuning data from a JSONL file."""

    data = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                data.append(json.loads(line))

    return data


def load_test_data(file_path):
    """Load test questions from a CSV file."""

    return pd.read_csv(file_path)


def validate_sft_data(data):
    """Validate that each training example has a question and answer."""

    required_fields = {"question", "answer"}

    for index, item in enumerate(data):
        if not required_fields.issubset(item.keys()):
            raise ValueError(
                f"Missing question or answer in example {index}"
            )

    return True
