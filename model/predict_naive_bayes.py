import joblib 
import numpy as np

# Load model
model_name = 'naive_bayes_ai_text_classifier'
path = f"results/{model_name}.pki"
model = joblib.load(path)

# Inference
def classify(text):
    probs = model.predict_proba([text])[0]
    label = np.argmax(probs)
    confidence = probs[label]
    return int(label), float(confidence)