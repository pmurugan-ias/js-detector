import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/prod_data.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")
X = model.encode(df["text"].tolist(), show_progress_bar=True)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred)) 
