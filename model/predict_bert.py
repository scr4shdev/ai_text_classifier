from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# Load tokenizer and model
model_name = "distilbert_ai_text_classifier"
path = f"results/{model_name}"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForSequenceClassification.from_pretrained(path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Inference
def classify(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {key: val.to(device) for key, val in inputs.items()} 
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
        return pred, confidence