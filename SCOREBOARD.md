# JUST Student Assistant — SCOREBOARD

## 1. Project Evaluation Overview

The current recovered evaluation compares:

- Qwen2.5-0.5B-Instruct Base
- Qwen2.5-0.5B-Instruct SFT

Evaluation dataset: 30 test questions.

Scoring system:
- 1.0 = Fully correct
- 0.5 = Partially correct
- 0.0 = Incorrect

The scores below are based on the existing manual review records.

## 2. Base vs. SFT Results

| Metric | Qwen Base | Qwen SFT |
|---|---:|---:|
| Total Questions | 30 | 30 |
| Weighted Score | 10.5 / 30 | 13.5 / 30 |
| Weighted Score (%) | 35% | 45% |
| Fully Correct | 6 | 7 |
| Partially Correct | 9 | 13 |
| Incorrect | 15 | 10 |
| Hallucinations | 13 | 8 |
| Incomplete Answers | 8 | 12 |
| Refusal Errors | 3 | 2 |
| Correct Refusals | 3 | 2 |

### Evaluation Observations

- Qwen SFT achieved a 10 percentage-point improvement
  over Qwen Base in weighted score.
- Fully correct answers increased from 6 to 7.
- Partially correct answers increased from 9 to 13.
- Incorrect answers decreased from 15 to 10.
- Recorded hallucinations decreased from 13 to 8.
- Refusal errors decreased from 3 to 2.

These observations describe the current reviewed test set.
They do not establish statistical significance or guarantee
performance on unseen questions.

### Evaluation Limitations

- The current review is not yet supported by a second
  independent human rater.
- Equivalent generation settings between Base and SFT
  must be verified before treating the comparison as final.
- The current evaluation summary script recalculates metrics
  from existing manually reviewed CSV files.
- It does not generate new model predictions.

## 3. GPT-2 CPT Training Results

The following results belong to the earlier GPT-2 CPT experiment.

| Metric | Result |
|---|---:|
| Model | GPT-2 |
| Training Stage | Continued Pre-Training |
| Training Documents | 34 |
| Sequence Length | 512 |
| Epochs | 1 |
| Training Steps | 28 |
| Initial Training Loss | 4.059410 |
| Final Training Loss | 2.594343 |
| Minimum Training Loss | 1.949719 |
| Average Training Loss | 2.975861 |
| Maximum Training Loss | 4.059410 |
| Loss Reduction | 1.465067 |

Notes:
- These metrics belong to GPT-2, not Qwen.
- The available loss records are training-loss records.
- Validation loss and CPT perplexity are not available
  in the currently recovered results.

## 4. Qwen CPT Status

- Qwen CPT training was performed in an earlier environment,
  according to the historical experiment records.
- Its adapter and verifiable training artifacts are not
  present in the currently recovered repository.
- Qwen CPT evaluation is not included in the current
  Base-versus-SFT comparison.
- No Qwen CPT metrics are claimed in this scoreboard.

## 5. Available Evaluation and Result Files

### Qwen Base Evaluation

- `results/model_results/qwen_base_30_predictions.csv`
- `results/model_results/qwen_base_30_evaluation_review.csv`
- `results/model_results/qwen_base_30_summary.csv`
- `results/model_results/qwen_base_summary_recalculated.csv`

### Qwen SFT Evaluation

- `data/tests/qwen_30_test_predictions_recovered.csv`
- `data/tests/qwen_30_evaluation_scored.csv`
- `results/model_results/qwen_30_evaluation_scored.csv`
- `results/model_results/qwen_sft_summary_recalculated.csv`

### Base vs. SFT Comparison

- `results/model_results/qwen_base_vs_sft_comparison.csv`
- `results/model_results/qwen_base_vs_sft_recalculated.csv`

### GPT-2 CPT Results

- `results/model_results/cpt_loss.csv`
- `results/model_results/gpt2_cpt_summary.csv`

## 6. Training and Evaluation Scripts

| Script | Purpose | Current Status |
|---|---|---|
| `train_cpt.py` | CPT training | Existing script; requires CUDA |
| `train_sft.py` | Qwen SFT with LoRA | Prepared and syntax-checked; new training run not performed |
| `score.py` | Calculate evaluation summaries | Tested on existing manual review files |

The evaluation script does not perform model inference,
latency measurement, or independent human scoring.

## 7. Pending Work

1. Obtain a second independent human evaluation of the
   30 test questions.
2. Verify equivalent generation settings for Base and SFT.
3. Recover and verify Qwen CPT adapters and training logs,
   if available.
4. Complete the required validation and perplexity analysis
   where supporting artifacts are available.
5. Measure and document average inference latency.
6. Verify the final repository contents and documentation.

## 8. Reporting Notes

- GPT-2 CPT results must not be presented as Qwen CPT results.
- Base and SFT scores are based on the current manual review.
- Recalculated CSV files contain summaries of existing scores;
  they are not a new model evaluation.
- Missing artifacts and pending evaluation requirements
  are explicitly identified rather than estimated.

---

This scoreboard reports the results and artifacts currently
available in the recovered project repository.
