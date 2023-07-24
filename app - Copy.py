import streamlit as st
import pandas as pd
import joblib
from model import title_model,v_headline,headline_model,v_title,prepos
import re
import datetime
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from textblob import TextBlob




# Load the trained models
model_title = joblib.load('pred_title.joblib')
model_headline = joblib.load('preds_headline.joblib')

# Load the vectorizers
vectorizer_title = joblib.load('vector_title.joblib')
vectorizer_headline = joblib.load('vector_headline.joblib')

test['Text_Title'] = test['Title'] + ' ' + test['Source'] + ' ' + test['Topic']


test['Text_Headline'] = test['Headline'] + ' ' + test['Source'] + ' ' + test['Topic']

test['Text_Title'] = [prepos_pll (x) for x in test['Text_Title']]


test['Text_Headline'] = [prepos_pll (x) for x in test['Text_Headline']]



vectorizer_title = TfidfVectorizer (use_idf=True)

test_v_Title = vectorizer_title.transform (test['Text_Title'])

vectorizer_headline = TfidfVectorizer ()


test_v_Headline = vectorizer_headline.transform (test['Text_Headline'])


test['polarity_t'] = test['Title'].apply (lambda x: TextBlob (x).sentiment.polarity)


test['subjectivity_t'] = test['Title'].apply (lambda x: TextBlob (x).sentiment.subjectivity)

test['polarity_h'] = test['Headline'].apply (lambda x: TextBlob (x).sentiment.polarity)


test['subjectivity_h'] = test['Headline'].apply (lambda x: TextBlob (x).sentiment.subjectivity)

encoder = LabelEncoder ()

test['Topic'] = encoder.transform (test['Topic'])

test['Source'] = encoder.transform (test['Source'])


test["no_word"] = test["Text_Title"].apply (lambda x: len (str (x).split ()))


test["no_un_word"] = test["Text_Title"].apply (lambda x: len (set (str (x).split ())))


test["no_char"] = test["Text_Title"].apply (lambda x: len (str (x)))


test["avg_word_len"] = test["Text_Title"].apply (lambda x: np.mean ([len (w) for w in str (x).split ()]))


test["no2_word"] = test["Text_Headline"].apply (lambda x: len (str (x).split ()))


test["no_un_word2"] = test["Text_Headline"].apply (lambda x: len (set (str (x).split ())))


test["no_char_2"] = test["Text_Headline"].apply (lambda x: len (str (x)))


test["avg_word_len_2"] = test["Text_Headline"].apply (lambda x: np.mean ([len (w) for w in str (x).split ()]))

scaler = StandardScaler ()

cols = ['Source', 'Topic', 'Facebook', 'GooglePlus', 'LinkedIn', 'no_word', 'no_un_word', 'no_char', 'avg_word_len',
        'no2_word', 'no_un_word2', 'no_char_2', 'avg_word_len_2']

for col in cols:
    test[col] = scaler.transform (test[col].values.reshape (-1, 1))

cols_t = ['Source', 'Topic', 'Facebook', 'GooglePlus', 'LinkedIn', 'no_word', 'no_un_word', 'no_char', 'avg_word_len',
          'no_word', 'no_un_word', 'no_char', 'avg_word_len', 'polarity_t', 'subjectivity_t']


test_X1 = test[cols_t]

cols_h = ['Source', 'Topic', 'Facebook', 'GooglePlus', 'LinkedIn', 'no_word', 'no_un_word', 'no_char', 'avg_word_len',
          'no2_word', 'no_un_word2', 'no_char_2', 'avg_word_len_2', 'polarity_h', 'subjectivity_h']

test_X2 = test[cols_h]

test_X_Title = hstack ([test_v_Title, csr_matrix (test_X1.values)])



test_X_Headline = hstack ([test_v_Headline, csr_matrix (test_X2.values)])


def main():
    st.title ("Sentiment Analysis")
    st.subheader ("Enter the text to analyze:")

    # Get user input
    st.write ("## Upload a CSV file")
    file = st.file_uploader ("Upload a CSV file", type="csv")
    if file is not None:
        test =pd.read_csv(file)
    # Perform sentiment analysis
    title_sentiment=model_title.predict()




    # Display the results
    st.subheader ("Sentiment Analysis Results:")
    st.write ("Title Sentiment: ", title_sent)
    st.write ("Headline Sentiment: ",headline_sent)


if __name__ == '__main__':
    main ()
