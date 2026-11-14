import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results" / "model_results"

BASE_REVIEW = RESULTS_DIR / "qwen_base_30_evaluation_review.csv"
SFT_REVIEW = RESULTS_DIR / "qwen_30_evaluation_scored.csv"
REFERENCE_FILE = PROJECT_DIR / "data" / "tests" / "test_reference_answers.jsonl"

REQUIRED_COLUMNS = {
    "question_number",
    "question",
    "predicted_answer",
    "reference_answer",
    "source",
    "score",
    "error_type",
    "review_notes",
}


def load_references():
    records = []
    with REFERENCE_FILE.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    if len(records) != 30:
        raise ValueError(f"Expected 30 reference answers, found {len(records)}")

    return records


def validate_review(path, references):
    if not path.exists():
        raise FileNotFoundError(f"Review file not found: {path}")

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    if len(df) != len(references):
        raise ValueError(
            f"{path.name}: expected {len(references)} rows, found {len(df)}"
        )

    for index, reference in enumerate(references):
        row = df.iloc[index]

        # Ignore formatting-only differences in question whitespace.
        csv_question = " ".join(str(row["question"]).split())
        json_question = " ".join(str(reference["question"]).split())

        if csv_question != json_question:
            raise ValueError(
                f"Question content mismatch in {path.name}, row {index + 1}"
            )

        # Ignore formatting-only differences in whitespace and line breaks.
        csv_reference = " ".join(str(row["reference_answer"]).split())
        json_reference = " ".join(str(reference["answer"]).split())

        if csv_reference != json_reference:
            raise ValueError(
                f"Reference answer content mismatch in {path.name}, row {index + 1}"
            )

    scores = pd.to_numeric(df["score"], errors="coerce")
    if scores.isna().any():
        raise ValueError(
            f"{path.name} has missing or invalid manual scores. "
            "Complete the review before calculating metrics."
        )

    if not scores.isin([0, 0.5, 1]).all():
        raise ValueError("Scores must be 0, 0.5, or 1.")

    df["score"] = scores
    return df


def calculate_summary(df, model_name):
    errors = df["error_type"].fillna("").astype(str).str.strip().str.lower()

    total = len(df)
    weighted_score = float(df["score"].sum())

    return {
        "Model": model_name,
        "Total Questions": total,
        "Weighted Score": weighted_score,
        "Weighted Score Percentage": round(weighted_score / total * 100, 2),
        "Fully Correct": int((df["score"] == 1).sum()),
        "Partially Correct": int((df["score"] == 0.5).sum()),
        "Incorrect": int((df["score"] == 0).sum()),
        "Hallucination": int(errors.eq("hallucination").sum()),
        "Incomplete Answer": int(
            errors.str.contains("incomplete", regex=False).sum()
        ),
        "Refusal Error": int(errors.str.contains("refusal error", regex=False).sum()),
        "Correct Refusal": int(errors.str.contains("correct refusal", regex=False).sum()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate metrics from manually reviewed model evaluations."
    )
    parser.add_argument("--base-review", default=str(BASE_REVIEW))
    parser.add_argument("--sft-review", default=str(SFT_REVIEW))
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    references = load_references()

    base_df = validate_review(Path(args.base_review), references)
    sft_df = validate_review(Path(args.sft_review), references)

    base_summary = calculate_summary(base_df, "Qwen2.5-0.5B-Instruct Base")
    sft_summary = calculate_summary(sft_df, "Qwen2.5-0.5B-Instruct SFT")

    base_output = RESULTS_DIR / "qwen_base_summary_recalculated.csv"
    sft_output = RESULTS_DIR / "qwen_sft_summary_recalculated.csv"
    comparison_output = RESULTS_DIR / "qwen_base_vs_sft_recalculated.csv"

    pd.DataFrame([base_summary]).to_csv(base_output, index=False)
    pd.DataFrame([sft_summary]).to_csv(sft_output, index=False)

    comparison = pd.DataFrame([
        {"Metric": key, "Qwen Base": base_summary[key], "Qwen SFT": sft_summary[key]}
        for key in [
            "Total Questions",
            "Weighted Score",
            "Weighted Score Percentage",
            "Fully Correct",
            "Partially Correct",
            "Incorrect",
            "Hallucination",
            "Incomplete Answer",
            "Refusal Error",
            "Correct Refusal",
        ]
    ])

    comparison.to_csv(comparison_output, index=False)

    print("Evaluation summaries recalculated successfully.")
    print("\nBASE:", base_summary)
    print("\nSFT:", sft_summary)
    print("\nComparison saved:", comparison_output)
    print("\nNote: These metrics use existing manual scores.")
    print("No model inference, CPT evaluation, latency, or independent second-rater scoring was performed.")


if __name__ == "__main__":
    main()
