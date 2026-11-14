# JUST Student Assistant — SCOREBOARD

## 1. Model Evaluation Results

Evaluation dataset: 30 test questions

Scoring system:
- 1.0 = Fully correct
- 0.5 = Partially correct
- 0.0 = Incorrect

| Metric | Qwen Base | Qwen SFT |
|---|---:|---:|
| Total Questions | 30 | 30 |
| Weighted Score | 10.5 / 30 | 13.5 / 30 |
| Weighted Score (%) | 35% | 45% |
| Fully Correct | 6 | 7 |
| Partially Correct | 9 | 13 |
| Incorrect | 15 | 10 |
| Hallucinations | 13 | 8 |
| Refusal Errors | 3 | 2 |

### Evaluation Observations

- Qwen SFT achieved a 10 percentage-point improvement
  over Qwen Base in weighted score.
- Hallucinations decreased from 13 to 8.
- Incorrect answers decreased from 15 to 10.
- These results are based on the current manual review.
- A second independent human rater is still required.
- Generation-setting equivalence between Base and SFT
  must be verified before treating the comparison as final.

## 2. CPT Training Results — GPT-2

| Metric | Result |
|---|---:|
| Model | GPT-2 |
| Training Stage | Continued Pre-Training |
| Epochs | 1 |
| Training Steps | 28 |
| Initial Training Loss | 4.059410 |
| Final Training Loss | 2.594343 |
| Minimum Training Loss | 1.949719 |
| Average Training Loss | 2.975861 |
| Maximum Training Loss | 4.059410 |
| Loss Reduction | 1.465067 |

Notes:
- These CPT results belong to the earlier GPT-2 experiment.
- They must not be presented as Qwen CPT results.
- Validation loss and CPT perplexity results are not
  available in the currently recovered results.

## 3. Qwen CPT Status

- Qwen CPT training artifacts and a verifiable training
  history are not present in the currently recovered
  results directory.
- Qwen CPT evaluation is not included in this scoreboard.
- No Qwen CPT metrics are claimed here.

## 4. Evaluation Limitations and Pending Work

1. Obtain an independent second human evaluation.
2. Verify that Base and SFT used equivalent generation settings.
3. Recover and verify Qwen CPT training artifacts and logs,
   if available.
4. Complete the required perplexity and validation analysis
   where the necessary artifacts are available.
5. Review and approve all project files before final GitHub push.

## 5. Result Files

### Qwen Base Evaluation
- results/model_results/qwen_base_30_predictions.csv
- results/model_results/qwen_base_30_evaluation_review.csv
- results/model_results/qwen_base_30_summary.csv

### Qwen SFT Evaluation
- data/tests/qwen_30_test_predictions_recovered.csv
- data/tests/qwen_30_evaluation_scored.csv
- results/model_results/qwen_30_evaluation_scored.csv

### Model Comparison
- results/model_results/qwen_base_vs_sft_comparison.csv

### CPT Results
- results/model_results/cpt_loss.csv
- results/model_results/gpt2_cpt_summary.csv

---
This scoreboard reports only results that are available
and identifiable in the recovered project files.
