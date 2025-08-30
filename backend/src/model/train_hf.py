#!/usr/bin/env python3
# train_hf.py — fine-tune DistilBERT (multi-class OK) on Google reviews
# Requires: pip install "transformers>=4.44" "datasets>=3.0.0" evaluate scikit-learn
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from typing import Dict
import numpy as np
import torch
import evaluate
import transformers
import datasets as hf_datasets  # for version print


    
from datasets import load_dataset, concatenate_datasets
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    set_seed,
)



print("TFX:", transformers.__version__, "| Datasets:", hf_datasets.__version__, "| Evaluate:", evaluate.__version__)

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
    "num_epochs": 6,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.06,
    "seed": 42,

    "output_dir": "checkpoints/distilbert_uncased",
    "early_stopping_patience": 2,
    "metric_for_best_model": "f1_macro",
    "manual_minority_weight": None,
}
# ==================================================


def main():
    cfg = CONFIG
    set_seed(cfg["seed"])

    # ---------- Load datasets (train/val/test) ----------
    ds = load_dataset("csv", data_files=cfg["data_files"])  # DatasetDict

    # ---------- Labels / mappings ----------
    with open(cfg["label2id_path"], "r") as f:
        label2id: Dict[str, int] = json.load(f)
    with open(cfg["id2label_path"], "r") as f:
        raw_id2label = json.load(f)
    id2label: Dict[int, str] = {int(k): v for k, v in raw_id2label.items()}
    num_labels = len(label2id)
    print(f"Detected classes ({num_labels}): {list(label2id.keys())}")

    # ---------- Tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)  # good for CUDA tensor cores

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg["max_length"],
        )

    ds = ds.map(tokenize, batched=True)

    # ---------- Ensure integer 'labels' column ----------
    if "label_id" in ds["train"].column_names:
        ds = ds.rename_column("label_id", "labels")
    else:
        def map_str_label_to_id(batch):
            return {"labels": [label2id[s] for s in batch["label"]]}
        ds = ds.map(map_str_label_to_id, batched=True)

    # ---------- Diagnostics (label counts) ----------
    import collections
    for split in ["train", "validation", "test"]:
        y = np.array(ds[split]["labels"])
        print("Label counts", split, ":", collections.Counter(y))

 
    y = np.array(ds["train"]["labels"])
    counts = np.bincount(y, minlength=num_labels)
    print("Train label counts:", collections.Counter(y))
    

    # ---------- Keep only model-needed columns ----------
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
        out = {
            "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
            "precision_macro": prec.compute(predictions=preds, references=labels, average="macro")["precision"],
            "recall_macro": rec.compute(predictions=preds, references=labels, average="macro")["recall"],
            "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
            "balanced_accuracy": balanced_accuracy_score(labels, preds),
            "mcc": matthews_corrcoef(labels, preds),
        }
        # For strictly binary case, add PR-AUC for minority:
        if logits.shape[1] == 2:
            # softmax
            probs = (logits - logits.max(axis=1, keepdims=True))
            probs = np.exp(probs) / np.exp(probs).sum(axis=1, keepdims=True)
            minority = int(np.bincount(labels).argmin())
            out["pr_auc_minority"] = average_precision_score(
                (labels == minority).astype(int), probs[:, minority]
            )
        return out

    # ---------- Class weights (inverse-frequency, multi-class OK) ----------
    y_train = np.array(ds["train"]["labels"])
    counts = np.bincount(y_train, minlength=num_labels)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    if cfg["manual_minority_weight"] is not None and num_labels == 2:
        minority = int(counts.argmin())
        weights = torch.tensor(
            [cfg["manual_minority_weight"] if i == minority else 1.0 for i in range(num_labels)],
            dtype=torch.float, device=device
        )
    else:
        # inverse-frequency for any number of classes
        weights = counts.sum() / np.maximum(counts, 1)
        weights = torch.tensor(weights, dtype=torch.float, device=device)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
            logits = outputs.get("logits")
            if model.training:
                loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # ---------- Training arguments ----------
    from torch import cuda, backends
    device_is_cuda = torch.cuda.is_available()
    if device_is_cuda:
        backends.cuda.matmul.allow_tf32 = True
        backends.cudnn.allow_tf32 = True

    args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        num_train_epochs=cfg["num_epochs"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        optim=("adamw_torch_fused" if device_is_cuda else "adamw_torch"),
        eval_strategy="epoch",       # <-- correct key
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=True,

        # Throughput
        dataloader_num_workers=4,          # try 4–8
        dataloader_pin_memory=device_is_cuda,  # True on CUDA, False on MPS/CPU

        # CUDA-only boosts:
        fp16=device_is_cuda,               # or bf16=True on Ampere+
        tf32=True if device_is_cuda else False,
        torch_compile=True if device_is_cuda else False,  # disable on MPS/CPU

        logging_steps=50,
        report_to=[],
        seed=cfg["seed"],
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
        data_collator=data_collator
    )

    # ---------- Train ----------
    print("\n>>> Starting training…")
    trainer.train()
    print(">>> Training complete.")

    # Save best model (+ tokenizer)
    print(">>> Saving best model & tokenizer…")
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

    # ---------- Hard-negative mining (one quick round) ----------
    # 1) Get validation predictions and find mistakes
    val_logits, val_labels, _ = trainer.predict(ds["validation"])
    val_preds = val_logits.argmax(axis=1)
    mist_idx = np.where(val_preds != val_labels)[0]
    print(f"\n[HNM] Hard negatives (val mistakes): {len(mist_idx)}")

    if len(mist_idx) > 0:
        # 2) Add misclassified validation rows back into TRAIN (with gold labels)
        hard_val = ds["validation"].select(mist_idx)
        ds["train"] = concatenate_datasets([ds["train"], hard_val]).shuffle(seed=42)

        # 3) Recompute class weights (no oversampling; rely on weights only)
        y_train = np.array(ds["train"]["labels"])
        counts = np.bincount(y_train, minlength=num_labels)
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

        if cfg.get("manual_minority_weight") is not None and num_labels == 2:
            minority = int(counts.argmin())
            weights = torch.tensor(
                [cfg["manual_minority_weight"] if i == minority else 1.0 for i in range(num_labels)],
                dtype=torch.float, device=device
            )
        else:
            # inverse-frequency weights for any number of classes
            weights = counts.sum() / np.maximum(counts, 1)
            weights = torch.tensor(weights, dtype=torch.float, device=device)
            # optional: normalize to mean 1 (cosmetic)
            weights = weights / weights.mean()

        # 4) OPTIONAL: shorter second round with lower LR
        trainer.args.num_train_epochs = 2
        trainer.args.learning_rate = float(cfg["learning_rate"]) * 0.5

        print("\n[HNM] Retraining briefly on augmented train (weights only, no oversampling)…")
        trainer.train(resume_from_checkpoint=True)

        # 5) Save again (overwrites best model with improved one)
        print("[HNM] Saving improved model…")
        trainer.save_model(cfg["output_dir"])
        tokenizer.save_pretrained(cfg["output_dir"])
    else:
        print("[HNM] No hard negatives found; skipping extra round.")



    # ---------- Threshold tuning (binary only) ----------
    if num_labels == 2:
        val_logits, val_labels, _ = trainer.predict(ds["validation"])
        probs_val = (val_logits - val_logits.max(axis=1, keepdims=True))
        probs_val = np.exp(probs_val) / np.exp(probs_val).sum(axis=1, keepdims=True)
        minority = int(np.bincount(val_labels).argmin())

        best_t, best_f1 = 0.5, -1.0
        from sklearn.metrics import f1_score
        for t in np.linspace(0.05, 0.95, 19):
            preds = (probs_val[:, minority] >= t).astype(int)
            y_true = (val_labels == minority).astype(int)
            f1_val = f1_score(y_true, preds, zero_division=0)
            if f1_val > best_f1:
                best_f1, best_t = f1_val, t
        print(f"Best threshold for minority ({id2label[minority]}): t={best_t:.2f}  F1={best_f1:.4f}")

        # Evaluate on test with tuned threshold
        test_logits, test_labels, _ = trainer.predict(ds["test"])
        probs_test = (test_logits - test_logits.max(axis=1, keepdims=True))
        probs_test = np.exp(probs_test) / np.exp(probs_test).sum(axis=1, keepdims=True)
        preds_thr = (probs_test[:, minority] >= best_t).astype(int)
        y_true = (test_labels == minority).astype(int)
        print("\n=== Thresholded (binary) report on TEST ===")
        print(classification_report(y_true, preds_thr,
                                    target_names=[f"not_{id2label[minority]}", id2label[minority]],
                                    digits=4))
        print("Confusion matrix (rows=true, cols=pred):")
        print(confusion_matrix(y_true, preds_thr))

    # ---------- Evaluate on test (standard argmax) ----------
    print("\n>>> Evaluating on test set…")
    test_metrics = trainer.evaluate(ds["test"])
    print("Test metrics:", test_metrics)

    logits, labels, _ = trainer.predict(ds["test"])
    preds = np.argmax(logits, axis=-1)
    target_names = [id2label[i] for i in range(num_labels)]
    print("\n=== Classification Report (Test, argmax) ===")
    print(classification_report(labels, preds, target_names=target_names, digits=4))
    print("=== Confusion Matrix (rows=true, cols=pred) ===")
    print(confusion_matrix(labels, preds))

    # ---------- Quick inference demo (edit texts) ----------
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); model.eval()
    demo_texts = [
        "Great service, very friendly staff!",                           # Good
        "Never going back here, worst service and long wait.",           # Rant
        "Best deals! 50% OFF — visit www.superdealz.com now!",           # Ad
        "asdfgh 😂😂😂😂 !!! http://a.co http://b.co",                   # Spam
    ]
    with torch.no_grad():
        for t in demo_texts:
            toks = tokenizer(t, return_tensors="pt", truncation=True, max_length=cfg["max_length"])
            toks = {k: v.to(device) for k, v in toks.items()}
            out = model(**toks)
            probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
            pred_id = int(probs.argmax())
            print(f"\nTEXT: {t}\nPRED: {id2label[pred_id]}  |  probs={ {id2label[i]: float(p) for i,p in enumerate(probs)} }")

if __name__ == "__main__":
    main()
