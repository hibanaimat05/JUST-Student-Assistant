# JUST Student Assistant

## 1. Project Overview

JUST Student Assistant is an AI-powered assistant designed to help students at Jordan University of Science and Technology (JUST) access university information.

The project combines Large Language Models (LLMs), model fine-tuning, and Retrieval-Augmented Generation (RAG).

## 2. Objectives

- Answer questions about JUST services.
- Build a university knowledge base.
- Apply CPT and SFT.
- Retrieve relevant information using RAG.
- Reduce unsupported answers.
- Evaluate answer quality.

## 3. Technologies

- Python
- PyTorch
- Hugging Face Transformers
- PEFT (LoRA)
- Qwen2.5-0.5B-Instruct
- Sentence Transformers
- Scikit-learn
- NumPy and Pandas

## 4. Model Training

### Base Model

Qwen/Qwen2.5-0.5B-Instruct

### Continued Pre-Training (CPT)

- 34 training text documents.
- One training epoch.
- LoRA adaptation.
- Adapter: results/qwen_cpt_model/checkpoint-epoch-1/

### Supervised Fine-Tuning (SFT)

- 304 valid question-answer records.
- 273 training records.
- 31 validation records.
- 3 epochs.
- LoRA fine-tuning.
- Answer-only loss.

Final validation loss: 2.2556858.

The validation loss measures training performance and does not directly represent factual accuracy.

## 5. Retrieval-Augmented Generation (RAG)

Implementation: src/rag_assistant.py

The RAG pipeline:
1. Encodes the student's question using multilingual embeddings.
2. Retrieves relevant university document chunks.
3. Calculates cosine similarity.
4. Rejects questions below the similarity threshold.
5. Extracts structured information when possible.
6. Uses the language model to generate context-based answers.

Configuration:
- Embedding model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- Chunk size: 700 characters.
- Chunk overlap: 120 characters.
- Top K: 3.
- Similarity threshold: 0.55.

Supported structured information includes phone numbers, fax numbers, emails, URLs, antivirus information, and housing application instructions.

### Initial RAG Tests

Six initial functional tests were performed:
- ITCC fax.
- ITCC phone.
- ITCC email.
- Antivirus software.
- Housing application instructions.
- Out-of-scope question.

Observed result: 6/6 tests passed.

This is a preliminary test, not a comprehensive performance benchmark.

## 6. Project Pipeline

1. Data collection.
2. Text extraction and cleaning.
3. Knowledge-base preparation.
4. CPT.
5. SFT dataset preparation.
6. SFT.
7. RAG implementation.
8. Evaluation and documentation.

## 7. Repository Structure

- train_cpt.py
- train_sft.py
- score.py
- config.yaml
- src/data_preparation.py
- src/model_loading.py
- src/chat_template.py
- src/rag_assistant.py
- data/cpt/
- data/sft_final.jsonl
- data/split_90_10/
- data/tests/
- results/qwen_cpt_model/
- results/qwen_sft_model/

## 8. Evaluation

Previously recorded Base-versus-SFT manual review:
- Qwen Base: 35%.
- Qwen SFT: 45%.

SFT training validation loss: 2.2556858.

These are different evaluation measures and should not be compared directly.

Limitations:
- The initial RAG test set is small.
- Independent second-rater review is recommended.
- Broader retrieval and answer-quality evaluation is still needed.
- Latency evaluation requires further documentation.

See SCOREBOARD.md for available evaluation records.

## 9. Running the Scripts

Run commands from the repository root.

Inspect training options:

    python train_cpt.py --help
    python train_sft.py --help

Run the evaluation summary:

    python score.py

Start training:

    python train_cpt.py
    python train_sft.py

Training requires the appropriate dependencies and a compatible GPU environment. Review config.yaml before running.

### RAG Module

The module is src/rag_assistant.py.

Main functions:
- configure_rag()
- retrieve()
- ask_just_assistant()
- just_assistant()

The embedding model, chunks, embeddings, tokenizer, and language model must be loaded and passed to configure_rag() before using the assistant.

## 10. Historical GPT-2 Experiments

Earlier CPT and SFT experiments used GPT-2. Historical checkpoint backups are stored separately in Google Drive under JUST_Student_Assistant_Backups.

These are historical experiments and are not the selected Qwen model.

## 11. Current Project Status

| Component | Status |
|---|---|
| Data preparation | Completed |
| CPT document split | Completed |
| Qwen CPT training | Completed |
| Qwen SFT training | Completed |
| CPT adapter | Saved |
| SFT adapter | Saved |
| RAG implementation | Completed |
| Structured extraction | Implemented |
| Initial RAG tests | 6/6 passed |
| Independent second-rater review | Recommended |
| Broader RAG evaluation | Pending |
| Final integration verification | Pending |

## 12. Institution

Jordan University of Science and Technology (JUST)
