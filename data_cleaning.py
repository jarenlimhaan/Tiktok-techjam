
import os, re, json
import pandas as pd
from sklearn.model_selection import train_test_split

# ========= CONFIG (edit if needed) =========
CONFIG = {
    "input_csv": "reviews.csv",      # path to your Kaggle CSV
    "outdir": "data/processed",      # where to write processed files
    "min_chars": 12,                 # drop reviews shorter than this (post-clean)
    "seed": 42,                      # RNG seed for reproducibility
    "splits": (0.8, 0.1, 0.1),       # train, val, test fractions
    "auto_label": True,              # generate weak policy labels automatically
    "violations_only": False,        # if True, drop 'Good' and keep only violations
    # Advanced: keep extra metadata columns in output CSVs for debugging/inspection
    "keep_extras": True,
}
# ==========================================

# --- regexes for light normalization (Transformer-friendly) ---
URL_RE   = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b")
MULTI_WS = re.compile(r"\s+")

def clean_text(s: str) -> str:
    """Lowercase, replace URL/EMAIL/PHONE with sentinels, normalize whitespace."""
    if not isinstance(s, str):
        return ""
    x = s.lower()
    x = URL_RE.sub(" <URL> ", x)
    x = EMAIL_RE.sub(" <EMAIL> ", x)
    x = PHONE_RE.sub(" <PHONE> ", x)
    x = MULTI_WS.sub(" ", x).strip()
    return x

# --- weak (heuristic) policy labels for quick start ---
AD_RE = re.compile(r"\b(deal|discount|promo|sale|use code|visit|order now|click here|subscribe|coupon)\b", re.I)
IRREL_RE = re.compile(r"(\bmy (?:new|old) (?:phone|laptop|car)\b|\boff-?topic\b|\bnot about (?:this|the) place\b)", re.I)
RANT_RE = re.compile(r"\b(never|not|haven't|didn'?t)\s+(?:been|visited)\b", re.I)

def weak_label(text: str) -> str:
    if RANT_RE.search(text): return "Rant"
    if URL_RE.search(text) or AD_RE.search(text): return "Ad"
    if IRREL_RE.search(text): return "Irrelevant"
    return "Good"

def main():
    cfg = CONFIG
    train_f, val_f, test_f = cfg["splits"]
    os.makedirs(cfg["outdir"], exist_ok=True)

    # Robust CSV read (quoted commas inside text are common)
    df = pd.read_csv(cfg["input_csv"], engine="python", on_bad_lines="skip")
    expected = {"business_name","author_name","text","photo","rating","rating_category"}
    missing = expected - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing columns: {missing}. Found: {list(df.columns)}")

    # Keep expected schema
    df = df[list(expected)].copy()

    # Clean text
    df["text"] = df["text"].apply(clean_text)

    # Drop empties / too short
    df = df[df["text"].str.len() >= cfg["min_chars"]].dropna(subset=["text"]).copy()

    # Deduplicate exact same text
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Labels
    if cfg["auto_label"]:
        df["label"] = df["text"].apply(weak_label)
        if cfg["violations_only"]:
            df = df[df["label"].isin(["Ad","Irrelevant","Rant"])].reset_index(drop=True)
    else:
        # If you prefer training on rating_category (taste/menu/...), uncomment next line:
        # df["label"] = df["rating_category"].astype(str)
        raise SystemExit("No policy labels present. Enable CONFIG['auto_label']=True or map your own labels.")

    # Build consistent id maps
    classes = sorted(df["label"].unique())
    label2id = {c:i for i,c in enumerate(classes)}
    id2label = {i:c for c,i in label2id.items()}
    df["label_id"] = df["label"].map(label2id)

    # Stratified splits
    strat = df["label"] if df["label"].nunique() > 1 else None
    train_df, tmp_df = train_test_split(df, test_size=(1 - train_f), random_state=cfg["seed"], stratify=strat)
    strat_tmp = tmp_df["label"] if tmp_df["label"].nunique() > 1 else None
    rel_val = val_f / (val_f + test_f)
    val_df, test_df = train_test_split(tmp_df, test_size=(1 - rel_val), random_state=cfg["seed"], stratify=strat_tmp)

    # Save minimal schema + optional extras
    def save(split_df, name):
        cols = ["text", "label", "label_id"]
        if cfg["keep_extras"]:
            cols += ["business_name","author_name","rating","rating_category","photo"]
        split_df[cols].to_csv(os.path.join(cfg["outdir"], f"{name}.csv"), index=False)

    save(train_df, "train")
    save(val_df,   "val")
    save(test_df,  "test")

    with open(os.path.join(cfg["outdir"], "label2id.json"), "w") as f:
        json.dump(label2id, f, indent=2)
    with open(os.path.join(cfg["outdir"], "id2label.json"), "w") as f:
        json.dump({int(k): v for k,v in id2label.items()}, f, indent=2)

    # Summary
    def dist(d): return d["label"].value_counts().to_dict()
    print("=== CLEAN SUMMARY ===")
    print(f"rows (post-clean): {len(df)}")
    print(f"classes: {classes} (label2id={label2id})")
    print(f"train: {len(train_df)}  dist: {dist(train_df)}")
    print(f"val:   {len(val_df)}    dist: {dist(val_df)}")
    print(f"test:  {len(test_df)}   dist: {dist(test_df)}")
    print(f"saved: {cfg['outdir']}/train.csv, val.csv, test.csv, label2id.json, id2label.json")

if __name__ == "__main__":
    main()
