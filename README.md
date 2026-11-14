# JUST Student Assistant

## 1. Project Overview

JUST Student Assistant is an AI-powered conversational assistant
designed to help students at Jordan University of Science and
Technology (JUST) access university-related information.

The project explores the use of Large Language Models (LLMs) to
answer student questions using information collected from
university-related resources.

The assistant aims to provide accessible, organized, and
source-grounded answers about university services and student
support resources.

## 2. Project Objectives

- Answer questions about JUST student services and resources.
- Organize university information into a usable knowledge base.
- Apply Continued Pre-Training (CPT) and Supervised Fine-Tuning (SFT).
- Evaluate model responses using a separate test-question set.
- Analyze answer quality, hallucinations, and refusal behavior.

## 3. Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face PEFT (LoRA)
- Pandas
- Scikit-learn
- YAML
- Large Language Models (LLMs)

## 4. Model and Training Approach

The project includes experiments with:

- GPT-2 for an earlier CPT experiment.
- Qwen2.5-0.5B-Instruct for the Base and SFT evaluation experiments.

### Continued Pre-Training (CPT)

CPT adapts a pre-trained language model using the collected
university-related text documents.

The recovered GPT-2 CPT experiment used:
- 34 training text documents.
- Sequence length of 512 tokens.
- One training epoch.
- 28 training steps.

The available GPT-2 training-loss results are documented in
`results/model_results/cpt_loss.csv`.

Qwen CPT training artifacts are not currently available in the
recovered repository, so no Qwen CPT metrics are claimed here.

### Supervised Fine-Tuning (SFT)

SFT uses question-answer pairs to teach the model how to respond
to student questions.

The prepared dataset contains 304 question-answer records.

The current `train_sft.py` script supports:
- Qwen2.5-0.5B-Instruct by default.
- LoRA fine-tuning.
- A 90% training and 10% validation split.
- Answer-only loss.
- Epoch-based evaluation and checkpoint saving.

The script has been prepared and syntax-checked. A new training
run has not been performed using this recovered script.

## 5. Project Pipeline

1. Data collection.
2. Document extraction and cleaning.
3. Knowledge-base preparation.
4. Training and validation data preparation.
5. Continued Pre-Training (CPT) experiments.
6. Supervised Fine-Tuning (SFT) preparation and experiments.
7. Model evaluation.
8. Results analysis and documentation.

## 6. Repository Structure

```text
JUST-Student-Assistant/
│
├── README.md
├── SCOREBOARD.md
├── config.yaml
├── train_cpt.py
├── train_sft.py
├── score.py
│
├── src/
│   ├── data_preparation.py
│   ├── model_loading.py
│   └── chat_template.py
│
├── data/
│   ├── cpt/
│   ├── final_dataset/
│   ├── split_90_10/
│   │   ├── train/
│   │   └── test/
│   ├── sft_final.jsonl
│   └── tests/
│
└── results/
    ├── loss_curves/
    └── model_results/
```

The structure above describes the main project directories.
Some experimental artifacts and model checkpoints may not be
present in the recovered repository.

## 7. Dataset Information

### CPT Data

The CPT data consists of university-related text documents
prepared for language-model adaptation.

The project includes cleaned text files and a 90/10 document
split into training and held-out test directories.

### SFT Data

The SFT dataset is stored in:

`data/sft_final.jsonl`

Each record contains:
- `question`
- `answer`

The dataset currently contains 304 valid question-answer records.

### Evaluation Data

The evaluation questions are stored in:

`data/tests/test_questions.csv`

The reference answers are stored in:

`data/tests/test_reference_answers.jsonl`

The evaluation set contains 30 questions.

## 8. Evaluation

The current Base-versus-SFT evaluation uses 30 manually reviewed
questions.

Scoring:
- 1.0 = Fully correct.
- 0.5 = Partially correct.
- 0.0 = Incorrect.

The current weighted scores are:
- Qwen Base: 35%.
- Qwen SFT: 45%.

These results indicate a 10 percentage-point improvement in
the current manual review.

Important evaluation limitations:
- A second independent human rater is still required.
- Equivalent generation settings between Base and SFT
  must be verified.
- Qwen CPT evaluation is not currently available.
- Latency and the required perplexity comparisons are not
  fully documented in the recovered results.

See `SCOREBOARD.md` for the available metrics and detailed
evaluation status.

## 9. Running the Scripts

Run commands from the repository root.

### Inspect SFT training options

```bash
python train_sft.py --help
```

The default SFT base model is:

`Qwen/Qwen2.5-0.5B-Instruct`

To train from a recovered CPT adapter, provide its directory
using `--cpt-adapter`. Do not provide this option unless the
adapter is actually available.

Training requires a suitable runtime and the required Python
packages. The script does not automatically provide a trained
model checkpoint.

### Run the evaluation summary script

```bash
python score.py
```

This script calculates summary metrics from the existing
manually reviewed Base and SFT evaluation CSV files.

It does not generate model predictions or perform new
model inference.

### CPT training

```bash
python train_cpt.py
```

The current CPT script requires CUDA. Review `config.yaml`
and the training settings before running it.

## 10. Results and Documentation

Important result files are stored in:

`results/model_results/`

These include:
- Base model predictions and evaluation review.
- SFT evaluation results.
- Base-versus-SFT comparison.
- GPT-2 CPT training-loss records.
- Recalculated evaluation summary CSV files.

The project scoreboard is maintained in:

`SCOREBOARD.md`

## 11. Current Project Status

| Component | Status |
|---|---|
| Data preparation | Completed |
| CPT document split | Completed |
| SFT dataset preparation | Completed |
| GPT-2 CPT experiment | Historical results available |
| Qwen Base evaluation | Available |
| Qwen SFT evaluation | Available |
| SFT training script | Prepared; new run not performed |
| Evaluation summary script | Prepared and tested on existing reviews |
| Qwen CPT artifacts | Not available in recovered repository |
| Independent second-rater review | Pending |
| Final evaluation verification | Pending |

## 12. Institution

Jordan University of Science and Technology (JUST)

---

This repository documents the available implementation,
experiments, evaluation results, and outstanding work.
Results are reported only when supporting files or records
are available in the recovered project.
