import streamlit as st
import pandas as pd
import pickle

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

data = pd.read_csv("sku_names.csv")

st.header("Matching distributors names :broccoli:")
sku = st.text_input('SKU name', " ")
button = st.button("Predict")

vectorizer_name = 'vectorizer.sav'
vectorizer_ = pickle.load(open(vectorizer_name, 'rb'))

if button:
    vectorize_sku = vectorizer_.transform([sku])
    output = loaded_model.predict(vectorize_sku)
    st.write(output[0])


st.caption('If you need to match list of names, you should make :green[__txt/csv__] file with :green[__UTF-8__] encoding and drop! :point_down:')
st.image("screen_1.png", caption="One SKU - one line on txt file!")

uploaded_file = st.file_uploader("Choose a file")


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('cp1251')


if uploaded_file:
    names_for_predict = pd.read_csv(uploaded_file, on_bad_lines='skip', header=None)
    st.success("Download success!")
    vectorize_names = vectorizer_.transform(names_for_predict[0])
    output_names = loaded_model.predict(vectorize_names)
    st.success("Match success!")

    st.caption("What's next? __Click for download!__")
    st.caption("Open file and follow the instructions:point_down:")

    result = pd.DataFrame(list(zip(names_for_predict[0], output_names)), columns=['distr_name', 'original_name'])
    csv = convert_df(result)
    st.download_button(
        label="Click for download!:point_left:",
        data=csv,
        file_name='result.csv',
        mime='text/csv',
    )
    st.image("screen_2.png", caption="Click on fisrt column and follow the instructions")



