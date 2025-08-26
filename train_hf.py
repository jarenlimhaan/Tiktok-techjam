
import os, json, numpy as np
from typing import Dict
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)
import transformers, datasets, evaluate, sklearn
from sklearn.metrics import classification_report, confusion_matrix

print("TFX:", transformers.__version__, "| Datasets:", datasets.__version__, "| Evaluate:", evaluate.__version__)

# ===================== CONFIG =====================
CONFIG = {
    "data_files": {
        "train": "data/processed/train.csv",
        "validation": "data/processed/val.csv",
        "test": "data/processed/test.csv",
    },
    "label2id_path": "data/processed/label2id.json",
    "id2label_path": "data/processed/id2label.json",

    "model_name": "distilbert-base-uncased",
    "max_length": 128,
    "batch_size": 16,
    "num_epochs": 4,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.06,
    "seed": 42,

    "output_dir": "checkpoints/distilbert_policy",
    "early_stopping_patience": 2,
    "metric_for_best_model": "f1_macro",
}
# ==================================================

def main():
    cfg = CONFIG
    set_seed(cfg["seed"])

    # ---------- Load datasets ----------
    ds = load_dataset("csv", data_files=cfg["data_files"])
    # Expect columns: 'text', 'label' (str) and/or 'label_id' (int), plus extras.

    # ---------- Labels / mappings ----------
    label2id: Dict[str, int] = json.load(open(cfg["label2id_path"]))
    raw_id2label = json.load(open(cfg["id2label_path"]))
    id2label: Dict[int, str] = {int(k): v for k, v in raw_id2label.items()}
    num_labels = len(label2id)
    print(f"Detected classes ({num_labels}): {list(label2id.keys())}")

    # ---------- Tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=cfg["max_length"],
        )

    ds = ds.map(tokenize, batched=True)

    # Ensure we have integer labels column named 'labels' for Trainer
    if "label_id" in ds["train"].column_names:
        ds = ds.rename_column("label_id", "labels")
    else:
        def map_str_label_to_id(batch):
            return {"labels": [label2id[s] for s in batch["label"]]}
        ds = ds.map(map_str_label_to_id, batched=True)

    # Keep only model-needed columns
    keep_cols = ["input_ids", "attention_mask", "labels"]
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep_cols])
    ds.set_format(type="torch", columns=keep_cols)

    # ---------- Model ----------
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # ---------- Metrics ----------
    acc = evaluate.load("accuracy")
    prec = evaluate.load("precision")
    rec = evaluate.load("recall")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
            "precision_macro": prec.compute(predictions=preds, references=labels, average="macro")["precision"],
            "recall_macro": rec.compute(predictions=preds, references=labels, average="macro")["recall"],
            "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

    # ---------- Class weights (handle imbalance) ----------
    y_train = np.array(ds["train"]["labels"])
    counts = np.bincount(y_train, minlength=num_labels)
    weights = counts.sum() / np.maximum(counts, 1)  # inverse freq
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    weights = torch.tensor(weights, dtype=torch.float, device=device)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
            logits = outputs.get("logits")
            # weighted during training; unweighted during eval for clearer val loss
            if model.training:
                loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # ---------- Training arguments ----------
    args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        num_train_epochs=cfg["num_epochs"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],

        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=True,

        logging_strategy="epoch",
        report_to=[],                 # safest way to disable wandb/tensorboard
        seed=cfg["seed"],
        dataloader_pin_memory=False,  # quiet MPS warning
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=cfg["early_stopping_patience"])]

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # ---------- Train ----------
    print("\n>>> Starting training…")
    trainer.train()
    print(">>> Training complete.")

    # Save best model (+ tokenizer)
    print(">>> Saving best model & tokenizer…")
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

    # ---------- Evaluate on test ----------
    print("\n>>> Evaluating on test set…")
    test_metrics = trainer.evaluate(ds["test"])
    print("Test metrics:", test_metrics)

    # Detailed classification report & confusion matrix
    logits, labels, _ = trainer.predict(ds["test"])
    preds = np.argmax(logits, axis=-1)
    target_names = [id2label[i] for i in range(num_labels)]
    print("\n=== Classification Report (Test) ===")
    print(classification_report(labels, preds, target_names=target_names, digits=4))
    print("=== Confusion Matrix (rows=true, cols=pred) ===")
    print(confusion_matrix(labels, preds))

    # ---------- Quick inference demo (device-safe) ----------
    model.to(device); model.eval()
    demo_texts = [
        "Never been here, but I heard it's terrible.",
        "Great service! Will come back again.",
        "Best prices! Visit www.superdealz.com for discounts",
        "Talking about my new phone here, not really this café.",
    ]
    with torch.no_grad():
        for t in demo_texts:
            toks = tokenizer(t, return_tensors="pt", truncation=True, max_length=cfg["max_length"])
            toks = {k: v.to(device) for k, v in toks.items()}
            out = model(**toks)
            pred_id = int(out.logits.argmax(dim=-1).cpu().numpy()[0])
            print(f"  {t}  ->  {id2label[pred_id]}")

if __name__ == "__main__":
    main()
