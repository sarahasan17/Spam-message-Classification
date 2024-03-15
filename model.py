import tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras.utils as ku
from wordcloud import WordCloud
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import joblib
import string
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv('spam.csv', encoding='latin-1')
def remove_punctuation(text):
    no_punct=[words for words in text if words not in string.punctuation]
    words_wo_punct=''.join(no_punct)
    return words_wo_punct
data['title_wo_punct']=data['v2'].apply(lambda x: remove_punctuation(x))

def tokenize(text):
    split=re.split("\W+",text)
    return split
data['title_wo_punct_split']=data['title_wo_punct'].apply(lambda x: tokenize(x.lower()))

stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    text=[word for word in text if word not in stopword]
    return text
data['title_wo_punct_split_wo_stopwords'] = data['title_wo_punct'].apply(lambda x: remove_stopwords(x))

ps=nltk.PorterStemmer()
def stemming(tokenized_text):
  text=[ps.stem(word) for word in tokenized_text]
  return text
data['title_wo_punct_split_wo_stemming']=data['title_wo_punct_split'].apply(lambda x: stemming(x))

wn=nltk.WordNetLemmatizer()
def lemmatizing(tokenized_text):
  text=[wn.lemmatize(word) for word in tokenized_text]
  return text
data['text_wo_lemmatizer']=data['title_wo_punct_split_wo_stemming'].apply(lambda x: lemmatizing(x))

c=0
data["new_val"] = len(data)*""
for i in data["text_wo_lemmatizer"]:
  l=""
  for j in i:
     if j.lower() not in stopwords.words('english') and j.isalpha():
      l+=j+" "
  data["new_val"][c]=l.strip()
  c+=1

vectorizer = CountVectorizer()
bow_transformer = vectorizer.fit(data['new_val'])

bow4 = bow_transformer.transform([data['new_val'][3]])
messages_bow = bow_transformer.transform(data['new_val'])
tfidf_transformer = TfidfTransformer().fit(messages_bow)
messages_tfidf = tfidf_transformer.transform(messages_bow)

vec = TfidfVectorizer(encoding = "latin-1", strip_accents = "unicode", stop_words = "english")
features = vec.fit_transform(data["new_val"])

msg_train, msg_test, label_train, label_test = \
train_test_split(messages_tfidf, data['v1'], test_size=0.2)

clf = MultinomialNB()
spam_detect_model = clf.fit(msg_train, label_train)

#predict_train = spam_detect_model.predict(msg_train)
#metrics.accuracy_score(label_train, predict_train)
joblib.dump(clf,"model.sav")








