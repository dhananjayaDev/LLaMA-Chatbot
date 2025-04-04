import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
with open("mangalam_qa_classifier_dataset.json", "r") as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Drop duplicate questions
df = df.drop_duplicates(subset="question")

# Features and labels
X = df["question"]
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize questions
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vect, y_train)

# Evaluate model
y_pred = model.predict(X_test_vect)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc * 100:.2f}%\n")
print("🔍 Classification Report:")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "business_question_classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n🎉 Model and vectorizer saved successfully.")
