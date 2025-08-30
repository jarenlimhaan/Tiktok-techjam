

import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==== Paths ====
MODEL_DIR = "checkpoints/distilbert_policy"
LABEL2ID_PATH = "data/processed/label2id.json"
ID2LABEL_PATH = "data/processed/id2label.json"

# ==== Load label maps ====
label2id = json.load(open(LABEL2ID_PATH))
raw_id2label = json.load(open(ID2LABEL_PATH))
id2label = {int(k): v for k, v in raw_id2label.items()}

# ==== Load model & tokenizer ====
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# Pick device automatically
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ==== Your custom test cases ====
texts = [
    "Nice people, good food and ambience",                           # Good
    "I hated the food, service was so bad and food took way too long to come",           # Rant
    "CHEAP CHEAP apartments at yishun!! Contact me at https//www.github.com",           # Ad
    "awoenweoiweicjwciwejwpedjwi3920e239rur34r83;lwemdlw",                   # Spam
]

# ==== Run predictions ====
with torch.no_grad():
    for t in texts:
        toks = tokenizer(t, return_tensors="pt", truncation=True, max_length=128, padding=True)
        toks = {k: v.to(device) for k, v in toks.items()}
        out = model(**toks)
        probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
        pred_id = int(probs.argmax())
        print(f"\nTEXT: {t}\nPRED: {id2label[pred_id]}  |  probs={ {id2label[i]: float(p) for i,p in enumerate(probs)} }")
