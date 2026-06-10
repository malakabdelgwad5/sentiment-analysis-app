import streamlit as st
import joblib

model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("Movie Review Sentiment Analysis")

review = st.text_area("Write a Review")

if st.button("Predict"):

    review_vector = vectorizer.transform([review])

    prediction = model.predict(review_vector)[0]

    if prediction == 1:
        st.success("Positive 😊")
    else:
        st.error("Negative 😔")