import argparse
import csv
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parent
BLOCK_SIZE = 512


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Connect to the GPU runtime.")

    torch.manual_seed(42)
    device = torch.device("cuda")

    config = yaml.safe_load((PROJECT_DIR / "config.yaml").read_text(encoding="utf-8"))
    train_dir = (PROJECT_DIR / config["paths"]["cpt_data"]).resolve()
    expected_dir = PROJECT_DIR / "data" / "split_90_10" / "train"
    if train_dir != expected_dir.resolve():
        raise ValueError(f"Expected the training split at {expected_dir}, got {train_dir}")

    files = sorted(path for path in train_dir.rglob("*.txt") if path.is_file())
    if not files:
        raise ValueError(f"No training TXT files found in {train_dir}")

    model_name = config["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    token_ids = []
    for path in files:
        text = path.read_text(encoding="utf-8-sig").strip()
        token_ids.extend(tokenizer.encode(text, add_special_tokens=False, verbose=False))
        token_ids.append(tokenizer.eos_token_id)

    usable_tokens = len(token_ids) // BLOCK_SIZE * BLOCK_SIZE
    if usable_tokens == 0:
        raise ValueError("Not enough tokens for a 512-token training sequence.")

    blocks = torch.tensor(token_ids[:usable_tokens], dtype=torch.long).reshape(-1, BLOCK_SIZE)
    batches = DataLoader(blocks, batch_size=1, shuffle=True)
    print(f"Model: {model_name} | TXT files: {len(files)} | Sequences: {len(blocks)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    output_dir = PROJECT_DIR / config["cpt"]["output_dir"]
    loss_dir = PROJECT_DIR / config["paths"]["loss_curves"]
    losses = []

    for epoch in range(1, args.epochs + 1):
        for step, input_ids in enumerate(batches, start=1):
            input_ids = input_ids.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
            loss_value = loss.item()
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, step {step}.")

            loss.backward()
            optimizer.step()
            losses.append((epoch, step, loss_value))

            if args.smoke_test:
                print(f"Smoke test passed: one optimizer step, loss={loss_value:.4f}", flush=True)
                return

            if step == 1 or step % 25 == 0 or step == len(batches):
                print(f"Epoch {epoch} | Step {step}/{len(batches)} | Loss {loss_value:.4f}", flush=True)

        checkpoint = output_dir / f"checkpoint-epoch-{epoch}"
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)

        loss_dir.mkdir(parents=True, exist_ok=True)
        with (loss_dir / "cpt_loss.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "step", "loss"])
            writer.writerows(losses)

        print(f"Saved checkpoint: {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
