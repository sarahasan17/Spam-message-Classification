import streamlit as st
from prediction import predict

# Set page title and favicon
st.set_page_config(page_title="SpamSnap", page_icon=":email:")

# Set app title and subtitle with custom fonts
st.title('SpamSnap')
st.markdown("<p style='font-size: 18px; color: #666666;'>A Spam Message Classification tool which can detect if a message is spam or not</p>", unsafe_allow_html=True)
# Add colorful background

text = st.text_area('Enter the message:)', 
                    height=100, 
                    key="text_input",
                    help="Enter a message to classify"
                   )

# Add colorful classify button
if st.button('Classify the message', 
             key="classify_button",
             help="Click to classify the message"
            ):
    if text.strip() == '':
        st.error('Please enter a message.')
    else:
        # Predict the class of the message
        res = predict([text])
        
        # Display the prediction result with colorful styling
        if res[0] == 'ham':
            st.success('This message is not spam.')
        else:
            st.error('This message is spam.')


