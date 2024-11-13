import streamlit as st
import numpy as np
import pandas as pd

st.header("My first Streamlit App")
st.write(pd.DataFrame({
    'Students': ['John', 'Lofa', 'Siti', 'Amy'],
    'Attendance Status': ['yes', 'yes', 'yes', 'no']
}))
