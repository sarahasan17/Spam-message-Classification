# SPAMSNAP: Spam Message Classifier
Spam Message Detection using NLP and TensorFlow

This repository contains code for building a spam message detection model using Natural Language Processing (NLP) techniques and TensorFlow. The model is trained on a dataset containing spam and non-spam (ham) messages.
Requirements
TensorFlow
NumPy
Pandas
Matplotlib
WordCloud
scikit-learn
NLTK

## Installation
To install TensorFlow, use the following command in bash:
!pip install tensorflow

## Clone the repository:
git clone https://github.com/sarahasan17/Spam-message-Classification.git

## Using streamlit
python model.py
python prediction.py
streamlit run app.py

## Description
The project consists of the following steps:

## Data Preprocessing:
Removal of punctuation from messages.
Tokenization of messages.
Removal of stopwords.
Stemming and lemmatization of words.

## Exploratory Data Analysis (EDA):
Analysis of spam and ham messages.
Identification of most used spam and ham words.

## Visualization:
Word Cloud representation of the most frequent words.

## Feature Engineering:
Bag of Words (BOW) representation.
Term Frequency-Inverse Document Frequency (TF-IDF) representation.

## Modeling:
Splitting the dataset into training and testing sets.
Training a Multinomial Naive Bayes classifier.
Evaluating the model's performance using accuracy, classification report, and confusion matrix.

## Conclusion:
Prediction of spam or ham label for new messages.

## Results
The model achieves an accuracy of approximately 97.2% on the test dataset.

## Credits
This project is inspired by various resources and tutorials on NLP and spam message detection.

## STREAMLIT SNIPPETS
![image](https://github.com/sarahasan17/SpamSnap/assets/103211125/7b571c0d-934c-4bdb-8323-13d83080801b)
<img width="953" alt="spam-1" src="https://github.com/sarahasan17/SpamSnap/assets/103211125/9366d8c6-d661-4ba8-bfdb-b353e971e283">

