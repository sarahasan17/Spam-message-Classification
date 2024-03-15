import streamlit as st
import pandas as pd
from prediction import predict

# Set page title and favicon
st.set_page_config(page_title="Spam Message Classification", page_icon=":email:")

# Set app title and subtitle
st.title('Spam Message Classification')
st.subheader('Detect if a message is spam or not')

# Add text input field for user input
text = st.text_area('Enter message', 'Type your message here...')

# Add button to classify the message
if st.button('Classify the message'):
    if text.strip() == '':
        st.error('Please enter a message.')
    else:
        # Predict the class of the message
        res = predict([text])
        
        # Display the prediction result
        if res[0] == 'ham':
            st.success('This message is not spam.')
        else:
            st.error('This message is spam.')

