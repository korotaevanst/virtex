import streamlit as st
import pandas as pd
import pickle

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

data = pd.read_csv("sku_names.csv")

st.header("App for :green[matching distributors names]:broccoli:")
with st.chat_message('ai'):
    st.write("Hello! Now, percentage of success - 95.1% :sparkles:")
sku = st.text_input('SKU name', " ")

vectorizer_name = 'vectorizer.sav'
vectorizer_ = pickle.load(open(vectorizer_name, 'rb'))

if sku:
    vectorize_sku = vectorizer_.transform([sku])
    output = loaded_model.predict(vectorize_sku)
    st.write(output[0])



st.caption('If you need to match list of names, you should make :green[__txt__] file with :green[__UTF-8__] encoding and drop! :point_down:')
st.image("screen_1.png", caption="One SKU - one line on txt file!")

uploaded_file = st.file_uploader("Choose a file")


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('cp1251')


def preprocessing(a):
    a = a.copy()
    a[0] = a[0].str.lower()
    a[0] = a[0].str.replace('"', ' ')
    a[0] = a[0].str.replace('*', ' ')
    a[0] = a[0].str.replace('/', ' ')
    a[0] = a[0].str.replace('шт', ' шт')
    a[0] = a[0].str.replace('гр', ' гр')
    a[0] = a[0].str.replace('0г', '0 г')
    return a


if uploaded_file:
    names_for_predict = pd.read_csv(uploaded_file, header=None, sep='delimiter')
    st.success("Download success!")
    vectorize_names = vectorizer_.transform(preprocessing(names_for_predict)[0])
    output_names = loaded_model.predict(vectorize_names)
    st.success("Match success!")

    result = pd.DataFrame(list(zip(names_for_predict[0], output_names)), columns=['distr_name', 'original_name'])
    st.dataframe(result)
    csv = convert_df(result)
    st.download_button(
        label="Click for download!:point_left:",
        data=csv,
        file_name='result.csv',
        mime='text/csv',
    )



