import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
import pickle

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

data = pd.read_csv("sku_names.csv")

st.header("Matching distributors names")
st.dataframe(data.head(5))
sku = st.text_input('SKU name for matching', "Жидкий дым")
button = st.button("Predict")

vectorizer_name = 'vectorizer.sav'
vectorizer_ = pickle.load(open(vectorizer_name, 'rb'))

if button:
    vectorize_sku = vectorizer_.transform([sku])
    output = loaded_model.predict(vectorize_sku)
    st.write(output[0])

uploaded_file = st.file_uploader("Choose a CSV file")

if uploaded_file:
    names_for_predict = pd.read_csv(uploaded_file)
    st.write(names_for_predict)
    vectorize_names = vectorizer_.transform(names_for_predict['names'])
    output_names = loaded_model.predict(vectorize_names)
    st.dataframe(output_names)


