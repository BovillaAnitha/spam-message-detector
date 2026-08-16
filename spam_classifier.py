import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

project_folder = os.path.dirname(__file__)

file_path = os.path.join(
    project_folder,
    "sms+spam+collection",
    "SMSSpamCollection"
)

df = pd.read_csv(
    file_path,
    sep="\t",
    header=None,
    names=["label", "message"]
)

print(df.head())
print(df.shape)


# ==========================================
# 2. CHECK DATA
# ==========================================

print("\nLabel counts:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())


# ==========================================
# 3. CONVERT LABELS
# ==========================================

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})


# ==========================================
# 4. SEPARATE INPUT AND OUTPUT
# ==========================================

X = df["message"]
y = df["label"]


# ==========================================
# 5. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==========================================
# 6. TF-IDF
# ==========================================
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTraining TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)


# ==========================================
# 7. CREATE AND TRAIN MODEL
# ==========================================

model = LogisticRegression()

model.fit(X_train_tfidf, y_train)

print("\nModel training completed!")


# ==========================================
# 8. NORMAL PREDICTIONS
# ==========================================

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)


# ==========================================
# 9. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["ham", "spam"]
    )
)


# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ==========================================
# 11. THRESHOLD COMPARISON
# ==========================================

spam_probabilities = model.predict_proba(X_test_tfidf)[:, 1]

thresholds = [0.50, 0.40, 0.30, 0.20]

print("\nThreshold comparison:")

for threshold in thresholds:

    predictions = (
        spam_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.2f} | "
        f"Recall: {recall:.2f} | "
        f"F1: {f1:.2f}"
    )


# ==========================================
# 12. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    os.path.join(project_folder, "spam_model.pkl")
)

joblib.dump(
    vectorizer,
    os.path.join(project_folder, "tfidf_vectorizer.pkl")
)

print("\nModel and vectorizer saved successfully!")