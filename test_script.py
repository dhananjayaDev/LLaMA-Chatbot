import joblib

# Load model and vectorizer
model = joblib.load("business_question_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Function to test input
def is_business_related(question):
    vec = vectorizer.transform([question])
    prediction = model.predict(vec)[0]
    return "Business-related" if prediction == 1 else "Unrelated"

# Try some questions
questions = [
    "Do you provide wedding photography services?",
    "How to boil an egg?",
    "Can I contact Mangalam vendors through the app?",
    "What is the tallest mountain in the world?"
]

for q in questions:
    print(f"Q: {q}\n➡️ {is_business_related(q)}\n")
