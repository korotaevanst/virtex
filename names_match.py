import streamlit as st
import pandas as pd
import pickle

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

data = pd.read_csv("sku_names.csv")

st.header("Matching distributors names")
st.dataframe(data.head(3))
sku = st.text_input('SKU name for matching', " ")
button = st.button("Predict")

vectorizer_name = 'vectorizer.sav'
vectorizer_ = pickle.load(open(vectorizer_name, 'rb'))

if button:
    vectorize_sku = vectorizer_.transform([sku])
    output = loaded_model.predict(vectorize_sku)
    st.write(output[0])

uploaded_file = st.file_uploader("Choose a file")


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


if uploaded_file:
    names_for_predict = pd.read_csv(uploaded_file, on_bad_lines='skip', header=None)
    st.write("Download success!")
    vectorize_names = vectorizer_.transform(names_for_predict[0])
    output_names = loaded_model.predict(vectorize_names)
    st.write("Match success!")

    result = pd.DataFrame(list(zip(names_for_predict[0], output_names)), columns=['distr_name', 'original_name'])
    csv = convert_df(result)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='large_df.csv',
        mime='text/csv',
    )



