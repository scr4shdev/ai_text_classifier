import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import joblib

df = pd.read_csv("data/AI_Human.csv")
df_human_generated = df[df["generated"] == 0]
df_ai_generated = df[df["generated"] == 1] 
n = min(len(df_human_generated), len(df_ai_generated))
df_balanced = pd.concat([
    df_human_generated.sample(n=n, random_state=42),
    df_ai_generated.sample(n=n, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)
print(df_balanced.head(10))
X = df_balanced["text"].tolist()
y = df_balanced["generated"].astype(int).tolist() 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ('count', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('train', MultinomialNB())
])

print("Training...")
model = pipeline.fit(X_train, y_train)
model_name = 'naive_bayes_ai_text_classifier'
path = f"results/{model_name}.pki"
joblib.dump(model, path)
y_preds = model.predict(X_test)
print(classification_report(y_test, y_preds, target_names=["Human", "AI"]))