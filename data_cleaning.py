#!/usr/bin/env python3
# data_cleaning_pseudolabel.py
# One-click cleaner + pseudo-labeler for "Ad" vs "Good" Google reviews.
# Input : reviews.csv with columns:
#         business_name, author_name, text, photo, rating, rating_category
# Output: data/processed/train.csv, val.csv, test.csv with columns: text,label,label_id,label_source
#         data/processed/label2id.json, id2label.json

import os, re, json, random
from typing import List, Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

# Hugging Face imports (pseudo-labeler)
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, pipeline
)

# ============== CONFIG ==============
CONFIG = {
    # Files
    "input_csv": "reviews.csv",
    "output_dir": "data/processed",
    "random_seed": 42,

    # Splits
    "val_size": 0.11,
    "test_size": 0.11,   # ~78/11/11 overall

    # Cleaning
    "min_chars": 20,
    "drop_dupes": True,

    # Label space (binary for now)
    "labels": ["Ad", "Good"],

    # Rule labeling sensitivity
    "ad_rules": {
        "lenient": True,   # allow softer combinations to count as Ad

        "check_urls": True,
        "check_contacts": True,
        "check_prices": True,
        "check_discounts": True,
        "check_cta": True,
        "check_keywords": True,
        "check_repetition": True,
        "check_caps_ratio": True,
        "check_emoji_row": True,

        "min_caps_ratio": 0.25,
        "min_repeated_word": 3,
        "min_emojis": 3,
    },

    # Pseudo-labeling
    # If a local fine-tuned checkpoint exists, we use it; else fall back to zero-shot.
    "teacher_model_dir": "checkpoints/distilbert_policy",  # your fine-tuned model (optional)
    "teacher_max_length": 128,

    # Thresholds to accept a pseudo-label (avoid noisy labels)
    "pseudo_threshold_overall": 0.90,   # top prob must be >= this
    "pseudo_threshold_ad": 0.60,        # if predicted "Ad", allow a lower bar to grow minority
    "min_margin": 0.20,                 # top_prob - second_prob must be >= this

    # Put pseudo-labeled rows ONLY into train (safer for evaluation)
    "pseudo_label_train_only": True,

    # Rebalance training to ~1:1 after labeling (oversample minority with replacement)
    "rebalance_train": True,
    "max_train_per_class": None,        # cap per class if you want (None = no cap)

    # Zero-shot model used if teacher not found (CPU fallback is fine)
    "zsl_model": "facebook/bart-large-mnli",
    "zsl_batch_size": 8,
}
# ====================================


# -------- regex library for rule Ad detection --------
URL_RE = re.compile(
    r"((https?://|www\.)\S+|[\w-]+\.(com|co|net|sg|io|me|info|biz|shop|store)\b)", re.I
)
PHONE_RE = re.compile(r"(\+?\d[\d\s\-]{7,}\d|\b\d{8}\b)", re.I)
EMAIL_RE = re.compile(r"\b[\w.\-+]+@[\w.\-]+\.[A-Za-z]{2,}\b", re.I)
SOCIAL_RE = re.compile(r"(instagram\.com/\w+|facebook\.com/\w+|t\.me/\w+|wa\.me/\w+|wechat|line id|tg\s*@\w+|ig\s*@\w+)", re.I)
CURRENCY_RE = re.compile(r"(\$+\s?\d+(?:[\.,]\d{2})?|\bsgd\b|s\$\s?\d+|rm\s?\d+|nt\$\s?\d+|usd\s?\d+)", re.I)
DISCOUNT_RE = re.compile(r"(\b\d{1,3}%\s?off\b|\bpromo(?:tion)?\b|\bcoupon\b|\buse\s+code\b|\bdeal\b|\bsale\b)", re.I)
CTA_RE = re.compile(r"(call\s+now|order\s+today|buy\s+now|dm\s+(us|me)|pm\s+us|click\s+here|subscribe|book\s+now|contact\s+(me|us))", re.I)
KEYWORDS_RE = re.compile(r"(free\s+delivery|free\s+install|cheap\s+price|best\s+price|limited\s*time|fast\s+deal|whatsapp)", re.I)
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")  # broad emoji ranges


def basic_clean(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def upper_ratio(tokens: List[str]) -> float:
    uppers = sum(1 for t in tokens if len(t) >= 2 and t.isupper())
    return 0.0 if not tokens else uppers / len(tokens)


def repeated_token_count(s: str) -> int:
    toks = re.findall(r"[A-Za-z0-9]+", s.lower())
    if not toks:
        return 0
    from collections import Counter
    c = Counter(toks)
    return max(c.values())


def rule_features(text: str, cfg: Dict) -> Dict[str, int]:
    feats = {k: 0 for k in ["url","contact","currency","discount","cta","keyword","repeat","caps","emoji"]}
    t = text
    if cfg["check_urls"] and URL_RE.search(t): feats["url"] += 1
    if cfg["check_contacts"] and (PHONE_RE.search(t) or EMAIL_RE.search(t) or SOCIAL_RE.search(t)): feats["contact"] += 1
    if cfg["check_prices"] and CURRENCY_RE.search(t): feats["currency"] += 1
    if cfg["check_discounts"] and DISCOUNT_RE.search(t): feats["discount"] += 1
    if cfg["check_cta"] and CTA_RE.search(t): feats["cta"] += 1
    if cfg["check_keywords"] and KEYWORDS_RE.search(t): feats["keyword"] += 1
    if cfg["check_repetition"] and repeated_token_count(t) >= cfg["min_repeated_word"]: feats["repeat"] += 1
    if cfg["check_caps_ratio"] and upper_ratio(t.split()) >= cfg["min_caps_ratio"]: feats["caps"] += 1
    if cfg["check_emoji_row"] and len(EMOJI_RE.findall(t)) >= cfg["min_emojis"]: feats["emoji"] += 1
    return feats


def rule_label(text: str) -> Tuple[str, str]:
    """
    Return (label, label_source)
    label_source: 'rule_strong', 'rule_weak', or '' (empty means undecided).
    """
    feats = rule_features(text, CONFIG["ad_rules"])
    hits = sum(1 for v in feats.values() if v > 0)

    # Strong Ad patterns
    strong_ad = (feats["url"] or feats["contact"] or feats["discount"] or feats["cta"])
    if strong_ad and hits >= 2:
        return "Ad", "rule_strong"

    # Lenient mode: softer combos count as Ad
    if CONFIG["ad_rules"]["lenient"]:
        if (feats["keyword"] and (feats["repeat"] or feats["caps"])) or (hits >= 2 and (feats["keyword"] or feats["currency"])):
            return "Ad", "rule_weak"

    # Strong Good: no signals at all
    if hits == 0:
        return "Good", "rule_strong"

    # Otherwise undecided
    return "", ""


# ----------------- PSEUDO-LABELER -----------------

def have_teacher_model() -> bool:
    d = CONFIG["teacher_model_dir"]
    return os.path.isdir(d) and os.path.isfile(os.path.join(d, "config.json"))

def device_name() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def pseudo_label_with_teacher(texts: List[str]) -> Tuple[List[str], List[float], List[float]]:
    """
    Use your fine-tuned classifier as teacher.
    Returns: labels, top_probs, margins
    """
    model_dir = CONFIG["teacher_model_dir"]
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
    dev = device_name()
    mdl.to(dev)
    mdl.eval()

    all_labels, all_top, all_margin = [], [], []
    bs = 32
    id2label = mdl.config.id2label
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            enc = tok(batch, truncation=True, padding=True,
                      max_length=CONFIG["teacher_max_length"], return_tensors="pt")
            enc = {k: v.to(dev) for k, v in enc.items()}
            logits = mdl(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=1)
            top = probs.max(axis=1)
            # margin = top - second best
            sorted_probs = -np.sort(-probs, axis=1)
            margin = sorted_probs[:,0] - sorted_probs[:,1]
            # map ids to string labels
            out_labels = [id2label[int(p)] for p in preds]
            all_labels.extend(out_labels)
            all_top.extend(top.tolist())
            all_margin.extend(margin.tolist())
    return all_labels, all_top, all_margin

def pseudo_label_with_zsl(texts: List[str]) -> Tuple[List[str], List[float], List[float]]:
    """
    Zero-shot fallback (CPU ok). Uses BART MNLI to choose between 'Ad' and 'Good'.
    """
    clf = pipeline("zero-shot-classification",
                   model=CONFIG["zsl_model"],
                   device=-1,             # CPU is safest across environments
                   truncation=True)
    labels = CONFIG["labels"]  # ["Ad","Good"]
    preds, tops, margins = [], [], []
    bs = CONFIG["zsl_batch_size"]
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        res = clf(batch, candidate_labels=labels, multi_label=False)
        # pipeline returns dict or list of dicts
        if isinstance(res, dict):
            res = [res]
        for r in res:
            seq_labels = r["labels"]
            seq_scores = r["scores"]
            # top & second
            top_label = seq_labels[0]
            top_prob = float(seq_scores[0])
            second_prob = float(seq_scores[1]) if len(seq_scores) > 1 else 0.0
            preds.append(top_label)
            tops.append(top_prob)
            margins.append(top_prob - second_prob)
    return preds, tops, margins


def apply_pseudo_labeling(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Runs rule labeling.
    - Pseudo-labels undecided rows with teacher or zero-shot.
    - Accepts only high-confidence pseudo labels.
    - If pseudo_label_train_only=True, pseudo-labeled rows go to TRAIN only (later step).
    """
    # 1) Rule labeling
    df = df.copy()
    labels, sources = [], []
    for t in df["text"].tolist():
        lab, src = rule_label(t)
        labels.append(lab)
        sources.append(src)
    df["label"] = labels
    df["label_source"] = sources

    # 2) Collect undecided rows
    undecided_mask = (df["label"] == "")
    undecided = df[undecided_mask]
    print(f"Undecided after rules: {len(undecided)}")

    if len(undecided) == 0:
        return df[df["label"] != ""].reset_index(drop=True)

    undecided_texts = undecided["text"].tolist()

    # 3) Pseudo-labeler
    use_teacher = have_teacher_model()
    print("Pseudo-labeler:", "teacher checkpoint" if use_teacher else "zero-shot (BART MNLI)")
    if use_teacher:
        pl_labels, pl_top, pl_margin = pseudo_label_with_teacher(undecided_texts)
    else:
        pl_labels, pl_top, pl_margin = pseudo_label_with_zsl(undecided_texts)

    # 4) Confidence filter
    AD = "Ad"
    THR_ALL = CONFIG["pseudo_threshold_overall"]
    THR_AD  = CONFIG["pseudo_threshold_ad"]
    THR_GAP = CONFIG["min_margin"]

    accept = []
    for lab, p, m in zip(pl_labels, pl_top, pl_margin):
        if lab == AD:
            ok = (p >= THR_AD) and (m >= THR_GAP)
        else:
            ok = (p >= THR_ALL) and (m >= THR_GAP)
        accept.append(ok)

    accepted_idx = undecided.index[accept]
    print(f"Pseudo-labeled accepted: {accept.count(True)} / {len(accept)}")

    # 5) Apply accepted pseudo labels
    df.loc[accepted_idx, "label"] = [pl_labels[i] for i, ok in enumerate(accept) if ok]
    df.loc[accepted_idx, "label_source"] = "pseudo_highconf"

    # Drop any rows still unlabeled
    df = df[df["label"] != ""].reset_index(drop=True)
    return df


# -------------- Rebalance helper --------------

def rebalance_train(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Make Ad:Good ~ 1:1 using oversample-with-replacement on minority.
    """
    counts = df_train["label"].value_counts()
    if set(counts.index) != set(CONFIG["labels"]):
        return df_train
    n_ad = int(counts.get("Ad", 0))
    n_good = int(counts.get("Good", 0))
    if n_ad == 0 or n_good == 0:
        return df_train

    target = max(n_ad, n_good)  # 1:1 target at the larger class
    rng = random.Random(CONFIG["random_seed"])

    def oversample(cls, need):
        part = df_train[df_train["label"] == cls]
        if need <= 0:
            return part
        extra = part.sample(n=need, replace=True, random_state=CONFIG["random_seed"])
        return pd.concat([part, extra], ignore_index=True)

    if n_ad < target:
        ad_part = oversample("Ad", target - n_ad)
        good_part = df_train[df_train["label"] == "Good"]
    else:
        good_part = oversample("Good", target - n_good)
        ad_part = df_train[df_train["label"] == "Ad"]

    out = pd.concat([ad_part, good_part], ignore_index=True).sample(frac=1.0, random_state=CONFIG["random_seed"])
    if CONFIG["max_train_per_class"]:
        out = (out.groupby("label", group_keys=False)
                 .head(CONFIG["max_train_per_class"]))
    return out


# -------------- Main pipeline --------------

def main():
    random.seed(CONFIG["random_seed"])
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # 1) Load CSV
    df = pd.read_csv(CONFIG["input_csv"])
    need = ["business_name", "author_name", "text", "photo", "rating", "rating_category"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing expected columns: {miss}")

    # 2) Clean
    df["text"] = df["text"].astype(str).map(basic_clean)
    df = df[df["text"].str.len() >= CONFIG["min_chars"]].copy()
    if CONFIG["drop_dupes"]:
        df = df.drop_duplicates(subset=["business_name", "text"]).reset_index(drop=True)

    # 3) Rules + Pseudo labeling
    df_labeled = apply_pseudo_labeling(df)

    # 4) Map labels to ids
    label2id = {lbl: i for i, lbl in enumerate(CONFIG["labels"])}
    id2label = {i: lbl for lbl, i in label2id.items()}
    df_labeled["label_id"] = df_labeled["label"].map(label2id)

    # 5) Split
    # Option A (safer): keep pseudo-labeled rows only in TRAIN
    if CONFIG["pseudo_label_train_only"]:
        train_part = df_labeled[df_labeled["label_source"] == "pseudo_highconf"]
        base_part  = df_labeled[df_labeled["label_source"] != "pseudo_highconf"]

        # split base_part into train/val/test (stratified)
        tr_base, tmp = train_test_split(
            base_part[["text","label","label_id","label_source"]],
            test_size=CONFIG["val_size"] + CONFIG["test_size"],
            random_state=CONFIG["random_seed"],
            stratify=base_part["label_id"],
        )
        rel_test = CONFIG["test_size"] / (CONFIG["val_size"] + CONFIG["test_size"])
        val_base, test_base = train_test_split(
            tmp, test_size=rel_test,
            random_state=CONFIG["random_seed"],
            stratify=tmp["label_id"],
        )
        # add pseudo-labeled rows to train only
        df_train = pd.concat([tr_base, train_part[["text","label","label_id","label_source"]]],
                             ignore_index=True).sample(frac=1.0, random_state=CONFIG["random_seed"])
        df_val = val_base.reset_index(drop=True)
        df_test = test_base.reset_index(drop=True)

    else:
        # Option B: stratified split on the full labeled set
        df_train, df_tmp = train_test_split(
            df_labeled[["text","label","label_id","label_source"]],
            test_size=CONFIG["val_size"] + CONFIG["test_size"],
            random_state=CONFIG["random_seed"],
            stratify=df_labeled["label_id"],
        )
        rel_test = CONFIG["test_size"] / (CONFIG["val_size"] + CONFIG["test_size"])
        df_val, df_test = train_test_split(
            df_tmp, test_size=rel_test,
            random_state=CONFIG["random_seed"],
            stratify=df_tmp["label_id"],
        )

    # 6) Rebalance TRAIN (optional)
    print("Label counts BEFORE rebalance:")
    print("  train:", df_train["label"].value_counts().to_dict())
    print("  val  :", df_val["label"].value_counts().to_dict())
    print("  test :", df_test["label"].value_counts().to_dict())

    if CONFIG["rebalance_train"]:
        df_train = rebalance_train(df_train)
        print("Label counts AFTER rebalance (train):", df_train["label"].value_counts().to_dict())

    # 7) Save processed files
    paths = {
        "train": os.path.join(CONFIG["output_dir"], "train.csv"),
        "val":   os.path.join(CONFIG["output_dir"], "val.csv"),
        "test":  os.path.join(CONFIG["output_dir"], "test.csv"),
    }
    df_train[["text","label","label_id"]].to_csv(paths["train"], index=False)
    df_val[["text","label","label_id"]].to_csv(paths["val"], index=False)
    df_test[["text","label","label_id"]].to_csv(paths["test"], index=False)

    with open(os.path.join(CONFIG["output_dir"], "label2id.json"), "w") as f:
        json.dump(label2id, f)
    with open(os.path.join(CONFIG["output_dir"], "id2label.json"), "w") as f:
        json.dump({str(k): v for k, v in id2label.items()}, f)

    print("\nWrote:")
    for k, p in paths.items():
        print(f"  {k}: {p} ({len(pd.read_csv(p))} rows)")
    print("  label maps:", os.path.join(CONFIG["output_dir"], "label2id.json"),
          os.path.join(CONFIG["output_dir"], "id2label.json"))

if __name__ == "__main__":
    # small numpy import needed by teacher margin calc
    import numpy as np  # noqa: F401
    main()
