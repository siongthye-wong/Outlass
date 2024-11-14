import streamlit as st
import pandas as pd
import seaborn as sns
import pickle

st.write("# Advertising Sales report")
st.write("This app predicts the **Advertising Sales** Value!")

st.sidebar.header('User Input Parameters')

def user_input_features():
    television = st.sidebar.slider('Television', 0.0, 150.0, 300.0)
    radio = st.sidebar.slider('Radio', 0.0, 150.0, 300.0)
    newspaper = st.sidebar.slider('Newspaper', 0.0, 150.0, 300.0)

    data = {'TV': television,
            'Radio': radio,
            'Newspaper': newspaper,}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)


modelSvr = pickle.load(open("modeladvertising.h5", "rb")) #rb: read binary
new_pred = modelSvr.predict(df) # testing (examination)

st.subheader('Prediction')
st.write(new_pred)

