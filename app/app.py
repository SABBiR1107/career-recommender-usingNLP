# app/app.py
import streamlit as st
import joblib
from utils.preprocess import clean_text, lemmatize
from utils.pdf_parser import extract_text_from_pdf

model = joblib.load("model/career_model.pkl")

st.set_page_config(page_title="Career Path Recommender")
st.title("🔍 NLP-Powered Career Path Recommender")

option = st.radio("Choose Input Method:", ["Text Input", "Upload PDF Resume"])

if option == "Text Input":
    text_input = st.text_area("Paste your resume, skills, or interests:", height=300)
    if st.button("Recommend Career"):
        if text_input:
            processed = lemmatize(clean_text(text_input))
            prediction = model.predict([processed])[0]
            st.success(f"🎯 Recommended Career Path: **{prediction}**")
        else:
            st.warning("Please enter some text.")

elif option == "Upload PDF Resume":
    uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])
    if uploaded_file:
        resume_text = extract_text_from_pdf(uploaded_file)
        processed = lemmatize(clean_text(resume_text))
        prediction = model.predict([processed])[0]
        st.success(f"📄 Predicted Career Path: **{prediction}**")
