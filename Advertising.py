import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.write("# Advertising Sales Report")
st.write("This web shows the total sales of different advertisting tool!")

st.sidebar.header('Input Parameters')

def user_input_features():
    television = st.sidebar.slider('Television', 4.3, 7.9, 5.4)
    newspapers = st.sidebar.slider('Newspapers', 2.0, 4.4, 3.4)
    radio = st.sidebar.slider('Radio', 1.0, 6.9, 1.3)
    data = {'Televeision': television,
            'Newspaper': newspapers,
            'Radio': radio,}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

