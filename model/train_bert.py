import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

from sklearn.metrics import classification_report

# 0. General
model_name = "distilbert-base-uncased"
trained_model_name = "distilbert_ai_text_classifier"

# 1. Loading the dataset
df = pd.read_csv("data/AI_Human.csv")
df_human_generated = df[df["generated"] == 0]
df_ai_generated = df[df["generated"] == 1] 
n = min(len(df_human_generated), len(df_ai_generated)) // 2
df_balanced = pd.concat([
    df_human_generated.sample(n=n, random_state=42),
    df_ai_generated.sample(n=n, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)
print(df_balanced.head(10))
X = df_balanced["text"].tolist()
y = df_balanced["generated"].astype(int).tolist() 

# 2. Train/test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 4. Dataset class
class AIDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        # tokenizing on every indexing to avoid creating large tensors and memory overflow
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        # squeeze to remove batch dimension from tokenizer output
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

train_dataset = AIDetectionDataset(X_train, y_train, tokenizer, max_length=256)
val_dataset = AIDetectionDataset(X_val, y_val, tokenizer, max_length=256)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 5. Load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2
).to(device)

# 6. Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    eval_strategy="steps",
    eval_steps = 200,
    warmup_steps=500,
    weight_decay=0.03,
    logging_dir="./logs",
    logging_steps=50,
    save_strategy="no",
    fp16=True,
    metric_for_best_model="eval_loss", 
    dataloader_num_workers=0,
    disable_tqdm = False,
    report_to=None,
)

# 7. Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
)

# 8. Train
print("Training...")
trainer.train()
path = f"results/{trained_model_name}"
trainer.save_model(path)
tokenizer.save_pretrained(path)


torch.cuda.empty_cache()

# 9. Evaluate
print("Evaluating...")
results = trainer.evaluate()
print(results)

predictions = trainer.predict(val_dataset)
preds = torch.argmax(torch.tensor(predictions.predictions), axis=1).numpy()
print(classification_report(y_val, preds, target_names=["Human", "AI"]))