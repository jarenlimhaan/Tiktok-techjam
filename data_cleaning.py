
import argparse, json, os, re
import pandas as pd
from sklearn.model_selection import train_test_split

# --- regexes for light normalization ---
URL_RE   = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b")
MULTI_WS = re.compile(r"\s+")

def clean_text(s: str) -> str:
    """Light, transformer-friendly cleanup: lowercase, replace URL/EMAIL/PHONE, normalize spaces."""
    if not isinstance(s, str):
        return ""
    x = s.lower()
    x = URL_RE.sub(" <URL> ", x)
    x = EMAIL_RE.sub(" <EMAIL> ", x)
    x = PHONE_RE.sub(" <PHONE> ", x)
    x = MULTI_WS.sub(" ", x).strip()
    return x

# --- weak (heuristic) labels so you can train now ---
AD_RE = re.compile(r"\b(deal|discount|promo|sale|use code|visit|order now|click here|subscribe|coupon)\b", re.I)
IRREL_HINTS = [
    r"\bmy (?:new|old) (?:phone|laptop|car)\b",
    r"\boff-?topic\b",
    r"\bnot about (?:this|the) place\b",
]
IRREL_RE = re.compile("|".join(IRREL_HINTS), re.I)
RANT_RE = re.compile(r"\b(never|not|haven't|didn'?t)\s+(?:been|visited)\b", re.I)

def weak_label(text: str) -> str:
    if RANT_RE.search(text): return "Rant"
    if URL_RE.search(text) or AD_RE.search(text): return "Ad"
    if IRREL_RE.search(text): return "Irrelevant"
    return "Good"

def main():
    ap = argparse.ArgumentParser(description="Clean Kaggle Google reviews and prepare splits for HF.")
    ap.add_argument("--input", default="reviews.csv", help="Path to Kaggle CSV")
    ap.add_argument("--outdir", default="data/processed", help="Where to write processed files")
    ap.add_argument("--min-chars", type=int, default=12, help="Drop reviews shorter than this after cleaning")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", default="0.8,0.1,0.1", help="train,val,test fractions")
    ap.add_argument("--auto-label", action="store_true",
                    help="Create weak policy labels: Ad/Irrelevant/Rant/Good")
    ap.add_argument("--violations-only", action="store_true",
                    help="When used with --auto-label, drop 'Good' examples")
    args = ap.parse_args()

    train_f, val_f, test_f = map(float, args.splits.split(","))
    os.makedirs(args.outdir, exist_ok=True)

    # Robust CSV read (some Kaggle CSVs have commas in text but are properly quoted)
    df = pd.read_csv(args.input, engine="python", on_bad_lines="skip")
    expected = {"business_name","author_name","text","photo","rating","rating_category"}
    missing = expected - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing columns: {missing}. Found: {list(df.columns)}")

    # Keep only the columns we care about; rename none (schema is fixed)
    df = df[list(expected)].copy()

    # Clean text
    df["text"] = df["text"].apply(clean_text)

    # Drop empties / too short
    df = df[df["text"].str.len() >= args.min_chars].dropna(subset=["text"]).copy()

    # Deduplicate exact same text (keeps first occurrence, preserves some metadata)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Labels:
    # - Your CSV has 'rating_category' (like 'taste','menu'), which is NOT our policy label.
    # - For policy training, use --auto-label to generate weak labels now.
    if args.auto_label:
        df["label"] = df["text"].apply(weak_label)
        if args.violations_only:
            df = df[df["label"].isin(["Ad","Irrelevant","Rant"])].reset_index(drop=True)
    else:
        # If you truly want to train on rating_category instead, uncomment next line:
        # df["label"] = df["rating_category"].astype(str)
        raise SystemExit("No policy labels present. Run with --auto-label to generate weak labels.")

    # Build consistent id maps
    classes = sorted(df["label"].unique())
    label2id = {c:i for i,c in enumerate(classes)}
    id2label = {i:c for c,i in label2id.items()}
    df["label_id"] = df["label"].map(label2id)

    # Split (stratify by label to keep class balance)
    strat = df["label"] if df["label"].nunique() > 1 else None
    train_df, tmp_df = train_test_split(df, test_size=(1 - train_f), random_state=args.seed, stratify=strat)
    strat_tmp = tmp_df["label"] if tmp_df["label"].nunique() > 1 else None
    rel_val = val_f / (val_f + test_f)
    val_df, test_df = train_test_split(tmp_df, test_size=(1 - rel_val), random_state=args.seed, stratify=strat_tmp)

    # Save minimal training schema + useful extras for debugging
    def save(split_df, name):
        # HF needs 'text' and either 'label' (str) or 'label_id' (int). Keep extras for reference.
        out = split_df[[
            "text", "label", "label_id",         # required / primary
            "business_name", "author_name",      # extras (helpful to inspect errors)
            "rating", "rating_category", "photo" # extras
        ]]
        out.to_csv(os.path.join(args.outdir, f"{name}.csv"), index=False)

    save(train_df, "train")
    save(val_df,   "val")
    save(test_df,  "test")

    with open(os.path.join(args.outdir, "label2id.json"), "w") as f:
        json.dump(label2id, f, indent=2)
    with open(os.path.join(args.outdir, "id2label.json"), "w") as f:
        json.dump({int(k): v for k,v in id2label.items()}, f, indent=2)

    # Summary
    def dist(d): return d["label"].value_counts().to_dict()
    print("=== CLEAN SUMMARY ===")
    print(f"rows (post-clean): {len(df)}")
    print(f"classes: {classes} (label2id={label2id})")
    print(f"train: {len(train_df)}  dist: {dist(train_df)}")
    print(f"val:   {len(val_df)}    dist: {dist(val_df)}")
    print(f"test:  {len(test_df)}   dist: {dist(test_df)}")
    print(f"saved to: {args.outdir}/train.csv, val.csv, test.csv + label2id.json, id2label.json")

if __name__ == "__main__":
    main()
