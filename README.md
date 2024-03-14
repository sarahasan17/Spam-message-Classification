# SPAM MESSAGE CLASSIFICATION
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

## Execute the Python script:
python spam_detection.py

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
The model achieves an accuracy of approximately X% on the test dataset.

## Credits
This project is inspired by various resources and tutorials on NLP and spam message detection.
