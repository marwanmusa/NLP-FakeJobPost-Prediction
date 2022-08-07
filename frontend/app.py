import streamlit as st
import requests
import json
import pandas as pd
from PIL import Image
import nltk
import re
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="Job Fake Prediction",
    page_icon="💼",
    initial_sidebar_state="collapsed",
    layout = "wide",
    menu_items={
        'Get Help': 'https://www.google.com/',
        'Report a bug': "https://github.com/marwanmusa",
        'About': "# Milestone 2 - Job Fake Prediction Application"
    }
)

col1, col2, col3 = st.columns([2,4,2])
with col2:
    image = Image.open('logo.png')
    st.image(image, use_column_width=True)

st.markdown("")
st.markdown("")
st.markdown("")
# image input input #
st.markdown("<h2 style='text-align: center; color: black;'>Input the Job Ads Description 💼</h2>", unsafe_allow_html=True)
with st.container():
    description1 = st.text_area("", placeholder="Paste here...")

submitted_text = st.button('submit')

if submitted_text:
    st.markdown("")
    st.markdown("")
    st.markdown("")    
    with st.container():
        st.markdown(f"### ***Your input is :*** \n {description1}")
        
    fraudulent = 1 # default value // not impacting the model result
    isidata = [description1, fraudulent]       
    columns = ['description', 'fraudulent']

    data_ = pd.DataFrame(data = [isidata], columns = columns)    

    # Menghilangkan kata-kata yang ada dalam list stopwords-english
    nltk.download('stopwords')

    # Fungsi untuk clean data
    def clean_text(text):
        '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
        and remove words containing numbers.'''
        text = str(text).lower() # Membuat text menjadi lower case
        text = re.sub('\[.*?\]', '', text) # Menghilangkan text dalam square brackets
        text = re.sub('https?://\S+|www\.\S+', '', text) # menghilangkan links
        text = re.sub('<.*?>+', '', text) # Menghilangkan text dalam <>
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # menghilangkan punctuatuion 
        text = re.sub('\n', '', text) # Menghilangkan enter / new line
        text = re.sub('\w*\d\w*', '', text) # Menghilangkan karakter yang terdiri dari huruf dan angka
        return text

    # cleaning data
    infdat = data_.drop('fraudulent', axis = 1)
    infdat['description'] = infdat['description'].apply(lambda x:clean_text(x))

    # Defining corpus with cleaned data
    ss = SnowballStemmer(language='english') 
    corpusinf = []
    for i in range(0, len(infdat)):
        decsr = infdat['description'][i]
        decsr = decsr.split()  # splitting data
        decsr = [ss.stem(word) for word in decsr if not word in stopwords.words('english')] # steeming setiap huruf dengan pengecualian kata yang ada dalam stopwords
        decsr = ' '.join(decsr)
        corpusinf.append(decsr)

    infdat['corpusinf'] = corpusinf
    infdat.reset_index(inplace = True)

    # encoding
    voc_size = 5000
    inf_enc_corps = [one_hot(words, voc_size) for words in corpusinf]

    # loading
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Tokenization
    descr_length = 40
    inf_word_idx = tokenizer.texts_to_sequences(infdat['corpusinf'])
    inf_padded_seqs = pad_sequences(inf_word_idx, maxlen = descr_length)

    input_data_json = json.dumps({
        "signature_name": "serving_default",
        "instances": inf_padded_seqs.tolist(),
    })

    URL = "http://fakejobprediction-app.herokuapp.com/v1/models/fake_job_prediction:predict"

    response = requests.post(URL, data=input_data_json)
    response.raise_for_status() # raise an exception in case of error
    response = response.json()

    # st.markdown("<h2 style='text-align: center; color: black;'>Customer's Data Recap</h2>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("### *& The Prediction is :*") 
    for res in response['predictions'][0]:
        if res > 0.5:
            st.markdown("<h2 style='text-align: center; color: red;'>Fake Job Ads</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='text-align: center; color: green;'>Real Job Ads</h2>", unsafe_allow_html=True)

