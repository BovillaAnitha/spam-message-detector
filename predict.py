import os
import joblib

# Get the folder where predict.py is located
project_folder = os.path.dirname(__file__)

# Load saved model and vectorizer
model = joblib.load(
    os.path.join(project_folder, "spam_model.pkl")
)

vectorizer = joblib.load(
    os.path.join(project_folder, "tfidf_vectorizer.pkl")
)

print("Spam Detector is ready!")

while True:

    message = input("\nEnter a message (type 'quit' to stop): ")

    if message.lower() == "quit":
        print("Program stopped.")
        break

    # Convert message to TF-IDF
    message_tfidf = vectorizer.transform([message])

    # Get spam probability
    probability = model.predict_proba(message_tfidf)[0][1]

    print(f"Spam probability: {probability:.2f}")

    # Our selected threshold
    if probability >= 0.30:
        print("🚨 SPAM")
    else:
        print("✅ NOT SPAM")