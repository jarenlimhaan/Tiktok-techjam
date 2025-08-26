from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ds = load_dataset("csv", data_files={
    "train": "data/processed/train.csv",
    "validation": "data/processed/val.csv",
    "test": "data/processed/test.csv",
})
label2id = json.load(open("data/processed/label2id.json"))
id2label = {int(k): v for k,v in json.load(open("data/processed/id2label.json")).items()}

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def tokenize(batch): return tok(batch["text"], padding=True, truncation=True, max_length=128)
ds = ds.map(tokenize, batched=True)
ds = ds.rename_column("label_id", "labels")  # Trainer expects 'labels'

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)
