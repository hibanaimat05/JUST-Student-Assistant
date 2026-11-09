import json
import pandas as pd


def load_sft_data(file_path):
    """Load SFT training data from JSONL."""

    data = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            item = json.loads(line)

            data.append(item)

    validate_sft_data(data)

    return data


def load_test_data(file_path):
    """Load test questions from CSV."""

    df = pd.read_csv(file_path)

    if "question" not in df.columns:
        raise ValueError("CSV must contain a question column")

    if df["question"].isnull().any():
        raise ValueError("Test questions contain empty values")

    return df


def validate_sft_data(data):
    """Validate SFT question-answer pairs."""

    required_fields = {"question", "answer"}

    if not data:
        raise ValueError("SFT dataset is empty")

    for index, item in enumerate(data):

        if not required_fields.issubset(item.keys()):
            raise ValueError(
                f"Missing question or answer in example {index}"
            )

        if not str(item["question"]).strip():
            raise ValueError(f"Empty question in example {index}")

        if not str(item["answer"]).strip():
            raise ValueError(f"Empty answer in example {index}")

    return True
