import argparse
import json
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class SFTDataset(Dataset):
    def __init__(self, records, tokenizer, max_length):
        self.items = []

        for item in records:
            messages = [
                {"role": "user", "content": str(item["question"]).strip()},
                {"role": "assistant", "content": str(item["answer"]).strip()},
            ]

            prompt_messages = messages[:-1]

            prompt_ids = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
            )

            full_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )

            full_ids = full_ids[:max_length]
            prompt_length = min(len(prompt_ids), len(full_ids))

            labels = [-100] * prompt_length + full_ids[prompt_length:]

            if not any(token != -100 for token in labels):
                continue

            self.items.append({
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            })

        if not self.items:
            raise ValueError("No usable SFT examples after tokenization.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class AnswerOnlyCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_length = max(len(x["input_ids"]) for x in features)
        pad_id = self.tokenizer.pad_token_id

        input_ids, attention_masks, labels = [], [], []

        for item in features:
            padding = max_length - len(item["input_ids"])

            input_ids.append(item["input_ids"] + [pad_id] * padding)
            attention_masks.append(item["attention_mask"] + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_records(path):
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            item = json.loads(line)

            if not str(item.get("question", "")).strip():
                raise ValueError(f"Empty question at line {line_number}")
            if not str(item.get("answer", "")).strip():
                raise ValueError(f"Empty answer at line {line_number}")

            records.append(item)

    if not records:
        raise ValueError("SFT dataset is empty.")

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--cpt-adapter", default=None)
    parser.add_argument("--epochs", type=int, choices=(2, 3), default=3)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "CUDA is unavailable. Training is disabled on CPU by default. "
            "Use --allow-cpu only if you intentionally want CPU training."
        )

    random.seed(42)
    torch.manual_seed(42)

    config = yaml.safe_load(
        (PROJECT_DIR / "config.yaml").read_text(encoding="utf-8")
    )

    data_path = PROJECT_DIR / config["paths"]["sft_data"]
    output_dir = PROJECT_DIR / config["sft"]["output_dir"]

    records = load_records(data_path)

    train_records, val_records = train_test_split(
        records,
        test_size=0.10,
        random_state=42,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer has no pad token or EOS token.")

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    if args.cpt_adapter:
        adapter_path = Path(args.cpt_adapter)
        if not adapter_path.exists():
            raise FileNotFoundError(f"CPT adapter not found: {adapter_path}")

        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            is_trainable=True,
        )
        print(f"Loaded CPT adapter: {adapter_path}")
    else:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj"
            ],
        )
        model = get_peft_model(model, lora_config)
        print("Starting SFT from the base model (no CPT adapter supplied).")

    model.print_trainable_parameters()

    train_dataset = SFTDataset(
        train_records, tokenizer, args.max_length
    )
    val_dataset = SFTDataset(
        val_records, tokenizer, args.max_length
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=AnswerOnlyCollator(tokenizer),
    )

    print(
        f"Model: {args.model_name}\n"
        f"Training records: {len(train_records)}\n"
        f"Validation records: {len(val_records)}\n"
        f"Training examples after tokenization: {len(train_dataset)}\n"
        f"Validation examples after tokenization: {len(val_dataset)}\n"
        f"Output: {output_dir}",
        flush=True,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = trainer.evaluate()
    metrics_path = output_dir / "final_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"Training completed. Final metrics: {metrics}", flush=True)


if __name__ == "__main__":
    main()
