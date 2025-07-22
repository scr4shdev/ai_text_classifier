# 🧠 AI Text Classifier

A Flask-based REST API for detecting whether a given text is **AI-generated** or **Human-written**, supporting two models:

- **Naive Bayes**: Fast classical ML model using scikit-learn.
- **BERT**: State-of-the-art Transformer fine-tuned on a balanced dataset.

---

## 🔍 Features

- Supports both:
  - Naive Bayes: A classic, fast, and lightweight statistical model ideal for quick prototyping and baseline performance on text classification using traditional bag-of-words features.
  - DistilBERT model: A powerful transformer-based deep learning model that captures complex language patterns and context, providing higher accuracy at the cost of more computational resources. This project uses DistilBERT as it is more lightweight in comparison to BERT-base.
- Returns classification label and confidence score.
- Easy-to-use JSON API.
- Balanced training on `data/AI_Human.csv`.
- Training scripts provided for both models.

---

## ⚙️ Requirements

- Python 3.8 to 3.12 (higher versions may cause compatibility issues)
- pip

---

## 🚀 Installation & Setup

Run the following commands to set up your environment:

```bash
# Clone the repository
git clone https://github.com/scr4shdev/ai_text_classifier.git
cd ai_text_classifier

# Create and activate a virtual environment
python3 -m venv venv

# On Linux/macOS
source venv/bin/activate

# On Windows
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Download the dataset from: https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text and paste it in `data/` folder

---

## 🏋️‍♂️ Training the Models

### ⚙️ Training the Naive Bayes Model

```bash
python model/train_naive_bayes.py
```

- Saves the model to `results/naive_bayes_ai_text_classifier.pki`

### ⚙️ Training DistilBERT Model

```bash
python model/train_bert.py
```

- Fine-tunes DistilBERT using Hugging Face Trainer.
- Saves the model and tokenizer to `results/distilbert_ai_text_classifier/`.

---

## 🖥️ Running the Flask API

Ensure the trained models exist in the `results/` folder, then start the API:

```bash
python main.py
```

---

## 📦 API Usage

```bash
POST /classify
Content-Type: application/json
```

```json
{
  "model": "bert", // or "naive_bayes"
  "text": "Your input text here"
}
```

Response example:

```json
{
  "label": 1, // 0=Human, 1=AI
  "confidence": 0.92
}
```

---

## 💡 Notes

- The API returns a 400 error if model or text fields are missing or empty.

- Send longer paragraphs in JSON "text" field.

- Confidence scores indicate model certainty.

- GPU recommended for BERT training but not mandatory.

- Naive Bayes is lightweight and suitable for quick experiments.
