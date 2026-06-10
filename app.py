import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# App title
st.title("🎬 Movie Review Sentiment Analysis")

# Description
st.write(
    "Enter a movie review below and the model will predict whether the sentiment is Positive or Negative."
)

# Text input
review = st.text_area("✍️ Write a Review")

# Prediction button
if st.button("Predict"):

    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        review_vector = vectorizer.transform([review])

        prediction = model.predict(review_vector)[0]

        # Confidence score
        probabilities = model.predict_proba(review_vector)[0]
        confidence = max(probabilities) * 100

        # Display result
        if prediction == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😔 Negative Review")

        st.write(f"**Confidence Score:** {confidence:.2f}%")
