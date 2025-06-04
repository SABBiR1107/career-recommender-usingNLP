# utils/preprocess.py
import spacy
import re

nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text

def lemmatize(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop])
