#!/usr/bin/env python3
# data_cleaning_pseudolabel.py
# One-click cleaner + pseudo-labeler for Google reviews.
# Input : reviews.csv (+ reviews2.csv) with columns:
#         business_name, author_name, text, photo, rating, rating_category
# Output: data/processed/train.csv, val.csv, test.csv
#         data/processed/label2id.json, id2label.json

import os, re, json, random
from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

# Hugging Face imports
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ============== CONFIG ==============
CONFIG = {
    "output_dir": "data/processed",
    "random_seed": 42,
    "val_size": 0.11,
    "test_size": 0.11,
    "min_chars": 20,
    "drop_dupes": True,
    "labels": ["Ad", "Good", "Rant", "Spam"],

    "teacher_model_dir": "checkpoints/distilbert_policy",
    "teacher_max_length": 128,
    "pseudo_threshold_overall": 0.95,
    "pseudo_threshold_ad": 0.60,
    "min_margin": 0.15,
    "pseudo_label_train_only": True,
    "rebalance_train": True,
    "zsl_model": "facebook/bart-large-mnli",
    "zsl_batch_size": 8,
}

# ============== REGEX (only obvious patterns) ==============
URL_RE   = re.compile(r'(https?://\S+|www\.\S+)', re.I)
EMAIL_RE = re.compile(r'\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b', re.I)
PHONE_RE = re.compile(r'\b\d{8,}\b')
SOCIAL_RE= re.compile(r'(instagram\.com|facebook\.com|t\.me|wa\.me|wechat|line id|ig\s*@\w+)', re.I)

MULTI_URL_RE = re.compile(r'(https?://\S+.*?https?://\S+)', re.I)
KEYBOARD_MASH_RE = re.compile(r'\b(asdfgh|qwerty|zzzz+|xxxx+|!!!!+)\b', re.I)

RANT_RE = re.compile(r'(never going back|worst service|waste of money|terrible|awful|horrible|disgusting|rude staff|long wait|dirty|scam)', re.I)

# ============== HELPERS ==============
def basic_clean(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()

# ============== RULE LABEL (minimal) ==============
def rule_label(text: str) -> Tuple[str, str]:
    """
    Minimal rules:
      - Only obvious Ads (URL, phone, socials).
      - Only obvious Spam (multi-links, nonsense mashes).
      - Only obvious Rants (clear rant phrases).
      - Everything else -> undecided (transformer decides).
    """
    if not text or not text.strip():
        return "", "empty"

    if URL_RE.search(text) or EMAIL_RE.search(text) or PHONE_RE.search(text) or SOCIAL_RE.search(text):
        return "Ad", "rule_obvious_ad"
    if MULTI_URL_RE.search(text) or KEYBOARD_MASH_RE.search(text):
        return "Spam", "rule_obvious_spam"
    if RANT_RE.search(text):
        return "Rant", "rule_obvious_rant"

    return "", "undecided"

# ============== PSEUDO-LABELER ==============
def have_teacher_model() -> bool:
    d = CONFIG["teacher_model_dir"]
    return os.path.isdir(d) and os.path.isfile(os.path.join(d, "config.json"))

def device_name() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def pseudo_label_with_teacher(texts: List[str]):
    model_dir = CONFIG["teacher_model_dir"]
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
    dev = device_name(); mdl.to(dev); mdl.eval()

    all_labels, all_top, all_margin = [], [], []
    bs = 32
    id2label = mdl.config.id2label
    import numpy as np
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            enc = tok(batch, truncation=True, padding=True,
                      max_length=CONFIG["teacher_max_length"], return_tensors="pt").to(dev)
            logits = mdl(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=1)
            top = probs.max(axis=1)
            margin = np.sort(probs, axis=1)[:,-1] - np.sort(probs, axis=1)[:,-2]
            all_labels.extend([id2label[int(p)] for p in preds])
            all_top.extend(top.tolist()); all_margin.extend(margin.tolist())
    return all_labels, all_top, all_margin

def pseudo_label_with_zsl(texts: List[str]):
    clf = pipeline("zero-shot-classification", model=CONFIG["zsl_model"], device=-1)
    labels = CONFIG["labels"]
    preds, tops, margins = [], [], []
    for i in range(0, len(texts), CONFIG["zsl_batch_size"]):
        batch = texts[i:i+CONFIG["zsl_batch_size"]]
        res = clf(batch, candidate_labels=labels, multi_label=False)
        if isinstance(res, dict): res = [res]
        for r in res:
            preds.append(r["labels"][0])
            tops.append(float(r["scores"][0]))
            margins.append(float(r["scores"][0] - r["scores"][1]))
    return preds, tops, margins

def apply_pseudo_labeling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    labels, sources = [], []
    for t in df["text"].tolist():
        lab, src = rule_label(t)
        labels.append(lab)
        sources.append(src)
    df["label"] = labels
    df["label_source"] = sources

    # Collect undecided + harvest rows
    undecided_mask = (df["label"] == "")
    undecided = df[undecided_mask]
    print(f"Undecided (incl harvest) after rules: {len(undecided)}")

    if len(undecided) == 0:
        return df[df["label"] != ""].reset_index(drop=True)

    undecided_texts = undecided["text"].tolist()

    # Pseudo-labeler
    use_teacher = have_teacher_model()
    print("Pseudo-labeler:", "teacher checkpoint" if use_teacher else "zero-shot (BART MNLI)")
    if use_teacher:
        pl_labels, pl_top, pl_margin = pseudo_label_with_teacher(undecided_texts)
    else:
        pl_labels, pl_top, pl_margin = pseudo_label_with_zsl(undecided_texts)

    RANT, SPAM, AD = "Rant", "Spam", "Ad"
    THR_ALL = CONFIG["pseudo_threshold_overall"]     # e.g. 0.95
    THR_AD  = CONFIG["pseudo_threshold_ad"]          # e.g. 0.60
    THR_GAP = CONFIG["min_margin"]                   # e.g. 0.15

    accept_list = []
    for idx, (lab, p, m) in enumerate(zip(pl_labels, pl_top, pl_margin)):
        src = undecided.iloc[idx]["label_source"]

        # Harvest candidates get looser thresholds
        if "harvest_ad" in src and lab == AD:
            ok = (p >= 0.50) and (m >= 0.10)
        elif "harvest_spam" in src and lab == SPAM:
            ok = (p >= 0.50) and (m >= 0.10)
        elif lab == AD:
            ok = (p >= THR_AD) and (m >= THR_GAP)
        elif lab in (RANT, SPAM):
            ok = (p >= 0.80) and (m >= 0.15)
        else:
            ok = (p >= THR_ALL) and (m >= THR_GAP)

        accept_list.append(ok)

    accept = pd.Series(accept_list, index=undecided.index, dtype=bool)
    accepted_idx = accept[accept].index
    print(f"Pseudo-labeled accepted: {int(accept.sum())} / {len(accept)}")

    if len(accepted_idx) == 0:
        return df[df["label"] != ""].reset_index(drop=True)

    pl_series = pd.Series(pl_labels, index=undecided.index)
    df.loc[accepted_idx, "label"] = pl_series.loc[accepted_idx].values
    df.loc[accepted_idx, "label_source"] = "pseudo_highconf"

    df = df[df["label"] != ""].reset_index(drop=True)
    return df


# ============== REBALANCE ==============
def rebalance_train(df_train: pd.DataFrame) -> pd.DataFrame:
    counts = df_train["label"].value_counts()
    target = counts.max()
    parts = []
    for lbl in CONFIG["labels"]:
        part = df_train[df_train["label"] == lbl]
        if not part.empty:
            extra = part.sample(n=target - len(part), replace=True, random_state=CONFIG["random_seed"])
            parts.append(pd.concat([part, extra], ignore_index=True))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=CONFIG["random_seed"])

# ============== MAIN ==============
def main():
    random.seed(CONFIG["random_seed"])
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # 1) Load & merge CSVs
    df1 = pd.read_csv("reviews.csv")
    df2 = pd.read_csv("reviews2.csv")

    need = ["business_name","author_name","text","photo","rating","rating_category"]
    for c in need:
        if c not in df2.columns:
            df2[c] = ""

    df = pd.concat([df1, df2], ignore_index=True)

    # 2) Clean
    df["text"] = df["text"].astype(str).map(basic_clean)
    df = df[df["text"].str.len() >= CONFIG["min_chars"]].copy()
    if CONFIG["drop_dupes"]:
        df = df.drop_duplicates(subset=["business_name", "text"]).reset_index(drop=True)

    # 3) Rules + Pseudo labeling
    df_labeled = apply_pseudo_labeling(df)

    # 4) Map labels → ids
    label2id = {lbl: i for i, lbl in enumerate(CONFIG["labels"])}
    id2label = {i: lbl for lbl, i in label2id.items()}
    df_labeled["label_id"] = df_labeled["label"].map(label2id)

    # --- helper: can we stratify? (every class needs ≥2 samples) ---
    from collections import Counter
    def can_stratify_frame(frame: pd.DataFrame) -> bool:
        if len(frame) == 0:
            return False
        cnt = Counter(frame["label_id"])
        return (len(cnt) >= 2) and all(v >= 2 for v in cnt.values())

    # 5) Split
    if CONFIG["pseudo_label_train_only"]:
        train_part = df_labeled[df_labeled["label_source"] == "pseudo_highconf"]
        base_part  = df_labeled[df_labeled["label_source"] != "pseudo_highconf"]

        print("Base counts before splitting:", base_part["label"].value_counts().to_dict())

        test_frac = CONFIG["test_size"]
        val_frac  = CONFIG["val_size"]
        total_test_val = test_frac + val_frac
        rel_test = test_frac / total_test_val if total_test_val > 0 else 0.5

        if len(base_part) == 0:
            # Edge case: no human/rule-labeled base — take val/test from full labeled (non-stratified)
            print("⚠️ No base_part rows; using non-stratified split from all labeled rows for val/test.")
            tr_base, tmp = train_test_split(
                df_labeled[["text","label","label_id","label_source"]],
                test_size=total_test_val,
                random_state=CONFIG["random_seed"],
                stratify=None,
            )
            val_base, test_base = train_test_split(
                tmp, test_size=rel_test,
                random_state=CONFIG["random_seed"],
                stratify=None,
            )
        else:
            if can_stratify_frame(base_part):
                tr_base, tmp = train_test_split(
                    base_part[["text","label","label_id","label_source"]],
                    test_size=total_test_val,
                    random_state=CONFIG["random_seed"],
                    stratify=base_part["label_id"],
                )
                if can_stratify_frame(tmp):
                    val_base, test_base = train_test_split(
                        tmp, test_size=rel_test,
                        random_state=CONFIG["random_seed"],
                        stratify=tmp["label_id"],
                    )
                else:
                    print("⚠️ Fallback: tmp too imbalanced for stratify → non-stratified val/test.")
                    val_base, test_base = train_test_split(
                        tmp, test_size=rel_test,
                        random_state=CONFIG["random_seed"],
                        stratify=None,
                    )
            else:
                print("⚠️ Fallback: base_part too imbalanced for stratify → non-stratified splits.")
                tr_base, tmp = train_test_split(
                    base_part[["text","label","label_id","label_source"]],
                    test_size=total_test_val,
                    random_state=CONFIG["random_seed"],
                    stratify=None,
                )
                val_base, test_base = train_test_split(
                    tmp, test_size=rel_test,
                    random_state=CONFIG["random_seed"],
                    stratify=None,
                )

        # Add high-confidence pseudo-labeled ONLY to train
        df_train = pd.concat(
            [tr_base, train_part[["text","label","label_id","label_source"]]],
            ignore_index=True
        ).sample(frac=1.0, random_state=CONFIG["random_seed"])
        df_val  = val_base.reset_index(drop=True)
        df_test = test_base.reset_index(drop=True)

    else:
        # Stratify on the full labeled set when possible; else fallback
        print("Full labeled counts before splitting:", df_labeled["label"].value_counts().to_dict())
        test_frac = CONFIG["test_size"]
        val_frac  = CONFIG["val_size"]
        total_test_val = test_frac + val_frac
        rel_test = test_frac / total_test_val if total_test_val > 0 else 0.5

        if can_stratify_frame(df_labeled):
            df_train, tmp = train_test_split(
                df_labeled[["text","label","label_id","label_source"]],
                test_size=total_test_val,
                random_state=CONFIG["random_seed"],
                stratify=df_labeled["label_id"],
            )
            if can_stratify_frame(tmp):
                df_val, df_test = train_test_split(
                    tmp, test_size=rel_test,
                    random_state=CONFIG["random_seed"],
                    stratify=tmp["label_id"],
                )
            else:
                print("⚠️ Fallback: tmp too imbalanced for stratify → non-stratified val/test.")
                df_val, df_test = train_test_split(
                    tmp, test_size=rel_test,
                    random_state=CONFIG["random_seed"],
                    stratify=None,
                )
        else:
            print("⚠️ Fallback: df_labeled too imbalanced for stratify → non-stratified splits.")
            df_train, tmp = train_test_split(
                df_labeled[["text","label","label_id","label_source"]],
                test_size=total_test_val,
                random_state=CONFIG["random_seed"],
                stratify=None,
            )
            df_val, df_test = train_test_split(
                tmp, test_size=rel_test,
                random_state=CONFIG["random_seed"],
                stratify=None,
            )

        # Keep indices tidy
        df_train = df_train.reset_index(drop=True)
        df_val   = df_val.reset_index(drop=True)
        df_test  = df_test.reset_index(drop=True)

    # 6) Rebalance TRAIN (optional)
    print("Label counts BEFORE rebalance:")
    print("  train:", df_train["label"].value_counts().to_dict())
    print("  val  :", df_val["label"].value_counts().to_dict())
    print("  test :", df_test["label"].value_counts().to_dict())

    if CONFIG["rebalance_train"]:
        df_train = rebalance_train(df_train)
        print("Label counts AFTER rebalance (train):", df_train["label"].value_counts().to_dict())

    # 7) Save
    outdir = CONFIG["output_dir"]
    df_train[["text","label","label_id"]].to_csv(f"{outdir}/train.csv", index=False)
    df_val[["text","label","label_id"]].to_csv(f"{outdir}/val.csv", index=False)
    df_test[["text","label","label_id"]].to_csv(f"{outdir}/test.csv", index=False)

    with open(f"{outdir}/label2id.json", "w") as f:
        json.dump(label2id, f)
    with open(f"{outdir}/id2label.json", "w") as f:
        json.dump({str(k): v for k, v in id2label.items()}, f)

    print("\nWrote train/val/test + label maps.")

if __name__=="__main__": 
    main()


