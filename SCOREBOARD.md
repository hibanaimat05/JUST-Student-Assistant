# JUST Student Assistant — SCOREBOARD

## 1. Project Evaluation Overview

The current evaluation includes three Qwen2.5-0.5B-Instruct variants:

- Base
- Continued Pre-Training (CPT)
- Supervised Fine-Tuning (SFT)

Evaluation dataset: 30 test questions per model.
CPT scores are provisional AI-assisted assessments and require
human verification.

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

## 4. Qwen CPT Evaluation

| Metric | Qwen CPT |
|---|---:|
| Total Questions | 30 |
| Weighted Score | 9 / 30 |
| Weighted Score (%) | 30% |
| Fully Correct | 4 |
| Partially Correct | 10 |
| Incorrect | 16 |

Evaluation notes:
- CPT evaluation was completed using AI-assisted review.
- These scores are provisional and have not been verified
  by independent human raters.
- CPT results should not be treated as final human evaluation.
- The CPT manual review file contains the question-level scores.

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

### Qwen CPT Evaluation

- `results/model_results/cpt_manual_review.csv`

### Three-Model Predictions and Review

- `results/model_results/phase2_three_model_predictions.csv`
- `results/model_results/phase2_three_model_review.csv`
- `results/model_results/phase2_human_evaluation.xlsx`

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

1. Verify the provisional CPT scores through independent
   human evaluation.
2. Obtain a second independent human evaluation of the
   30 test questions.
3. Verify equivalent generation settings for Base, CPT, and SFT.
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


## 9. AI-Assisted Evaluation Suggestions (November 2026)

A separate question-level evaluation workbook was prepared
for the three Qwen2.5-0.5B-Instruct variants.

Evaluation dataset: 30 questions per model.

| Metric | Qwen Base | Qwen CPT | Qwen SFT |
|---|---:|---:|---:|
| Total Questions | 30 | 30 | 30 |
| AI-Assisted Suggested Score | 8 / 30 | 8.5 / 30 | 11.5 / 30 |
| Suggested Percentage | 26.67% | 28.33% | 38.33% |

Important notes:
- These scores are AI-assisted suggestions based on question-level
  comparisons with the reference answers.
- The project team has reviewed some of the suggested assessments.
- These scores are separate from the previously recorded evaluation
  results in Sections 2 and 4.
- They must not replace the existing manual-review results.
- Independent human verification and final score approval are pending.
- Reviewer 1, Reviewer 2, and Final Score fields remain unfilled
  in the suggestion workbook.

Workbook:
`results/model_results/phase2_human_evaluation_ALL_suggestions.xlsx`


## 10. Standardized Generation Settings

The three-model evaluation (Base, CPT, and SFT) used a shared
generation configuration in the evaluation notebook.

Generation settings:

| Parameter | Value |
|---|---|
| Base Model | Qwen/Qwen2.5-0.5B-Instruct |
| max_new_tokens | 150 |
| do_sample | False |
| pad_token_id | tokenizer.pad_token_id |
| eos_token_id | tokenizer.eos_token_id |
| Test Questions | 30 per model |
| Total Predictions | 90 |

Model variants:
- Base: Original pretrained model without an adapter.
- CPT: Base model with the CPT adapter.
- SFT: Base model with the SFT adapter.

All three variants were evaluated using the same
`model.generate()` parameters and the same test questions.

Scope limitation:

This verification applies to the three-model evaluation
that generated `phase2_three_model_predictions.csv`.
It does not independently establish the generation settings
used for earlier separate Base, SFT, or CPT evaluations.


## 11. Perplexity Evaluation Results

Perplexity evaluation was completed for the three Qwen2.5-0.5B-Instruct
model variants using 30 question-reference answer pairs.

### Evaluation Results

| Metric | Qwen Base | Qwen CPT | Qwen SFT |
|---|---:|---:|---:|
| Test Questions | 30 | 30 | 30 |
| Answer Tokens | 764 | 764 | 764 |
| Average NLL | 3.142745 | 3.031491 | 2.564734 |
| Perplexity | 23.167378 | 20.728115 | 12.997207 |

### Interpretation

- Qwen CPT achieved approximately 10.53% lower perplexity than Qwen Base.
- Qwen SFT achieved approximately 43.90% lower perplexity than Qwen Base.
- Qwen SFT achieved approximately 37.29% lower perplexity than Qwen CPT.

Qwen SFT obtained the lowest perplexity among the three evaluated models.

Lower perplexity indicates better prediction of the reference answer tokens
under the evaluation setup. It does not independently establish factual
correctness or overall response quality.

### Evaluation Method

- Base model: Qwen/Qwen2.5-0.5B-Instruct
- CPT: Base model with the CPT LoRA adapter.
- SFT: Base model with the SFT LoRA adapter.
- Evaluation examples: 30.
- Reference answer tokens: 764 per model.
- Only reference answer tokens were included in the loss calculation.
- The same questions and reference answers were used for all three models.

The independence of this evaluation dataset from all training and
fine-tuning data must be verified before describing it as an
independent held-out test set.

### Result File

- `results/model_results/phase2_perplexity_results.csv`

### Status

Perplexity calculation completed for Base, CPT, and SFT.
