import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict

st.title('Spam Message Classification')

text = st.text_area('Enter message',None)

if st.button('Classify the message'):
    res = predict([text])
    st.text(res[0])
