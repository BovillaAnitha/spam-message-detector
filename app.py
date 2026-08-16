import streamlit as st
import joblib
import os


# ==========================================
# 1. LOAD MODEL AND VECTORIZER
# ==========================================

project_folder = os.path.dirname(__file__)

model = joblib.load(
    os.path.join(project_folder, "spam_model.pkl")
)

vectorizer = joblib.load(
    os.path.join(project_folder, "tfidf_vectorizer.pkl")
)


# ==========================================
# 2. SUSPICIOUS PATTERN DETECTOR
# ==========================================

def detect_suspicious_patterns(message):

    message = message.lower()

    suspicious_patterns = [
        "credit card",
        "credit details",
        "card details",
        "bank details",
        "bank account",
        "account number",
        "otp",
        "one time password",
        "password",
        "pin",
        "cvv",
        "upi",
        "send money",
        "transfer money",
        "deposit money",
        "claim your prize",
        "you won",
        "winner",
        "lottery",
        "click here",
        "verify your account",
        "urgent payment"
    ]

    found_patterns = []

    for pattern in suspicious_patterns:

        if pattern in message:
            found_patterns.append(pattern)

    return found_patterns


# ==========================================
# 3. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="📱"
)


# ==========================================
# 4. TITLE
# ==========================================

st.title("📱 Spam Message Detector")

st.write(
    "Enter an SMS message below and our AI model "
    "will analyze whether it is spam, suspicious, or safe."
)


# ==========================================
# 5. MESSAGE INPUT
# ==========================================

message = st.text_area(
    "Enter your message:",
    height=150,
    placeholder="Example: Congratulations! You won a free prize!"
)


# ==========================================
# 6. CHECK MESSAGE
# ==========================================

if st.button("Check Message"):

    if message.strip() == "":

        st.warning("⚠️ Please enter a message.")

    else:

        # --------------------------------------
        # Convert message to TF-IDF
        # --------------------------------------

        message_tfidf = vectorizer.transform([message])


        # --------------------------------------
        # Get spam probability
        # --------------------------------------

        probability = model.predict_proba(
            message_tfidf
        )[0][1]


        # --------------------------------------
        # Spam threshold
        # --------------------------------------

        threshold = 0.30


        # --------------------------------------
        # Check suspicious patterns
        # --------------------------------------

        suspicious_patterns = detect_suspicious_patterns(
            message
        )


        # ======================================
        # 7. FINAL RESULT
        # ======================================

        if probability >= threshold:

            st.error("🚨 SPAM MESSAGE")

            st.write(
                "The AI model considers this message likely to be spam."
            )


        elif suspicious_patterns:

            st.warning("⚠️ SUSPICIOUS MESSAGE")

            st.write(
                "This message contains potentially risky "
                "language or requests."
            )

            st.write("Detected patterns:")

            for pattern in suspicious_patterns:

                st.write(f"• {pattern}")


            st.info(
                "🔒 Do not share passwords, OTPs, "
                "bank details, card information, or PINs."
            )


        else:

            st.success("✅ NOT SPAM")

            st.write(
                "No strong spam or suspicious patterns were detected."
            )


        # ======================================
        # 8. SHOW PROBABILITY
        # ======================================

        st.write(
            f"Spam probability: **{probability:.2%}**"
        )