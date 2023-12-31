# -*- coding: utf-8 -*-
"""Sentimentanalysis_News.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19mveiABEgDUQ2nyzhv8Et4m_KHwu4dBq
"""

#!pip install -r "/content/drive/MyDrive/OneApi/req.txt"

#from google.colab import drive
#drive.mount('/content/drive')

#from sklearnex import patch_sklearn
#patch_sklearn()

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from textblob import TextBlob
import nltk
import re
import datetime
import matplotlib.pyplot as plt
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVR
#import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
#from wordcloud import WordCloud, STOPWORDS
import joblib
import streamlit as st

# %matplotlib inline

#!pip install modin


from multiprocessing import Pool
import modin.pandas as md
import time

# Commented out IPython magic to ensure Python compatibility.
# %time
train = pd.read_csv("E:/oneapi/oneapi/oneapi/training_data.csv")
# %time
##train=md.read_csv("/content/drive/MyDrive/OneApi/training_data.csv")


st.write("# Sentiment Analysis of News")
st.write("## Upload a CSV file")
file = st.file_uploader("Upload a CSV file", type="csv")
if file is not None:
    test =pd.read_csv(file)
    time.sleep(5)
#submission = pd.read_csv('V:/PESU/SEM 6/oneapi/sample.csv')
test_id = test['IDLink']
#test.head()



#train.head()

#train.info()

#train.describe()



train.isnull().sum()

test.isnull().sum()

train['Source'].value_counts()[:5]

train['Source'] = train['Source'].fillna('Bloomberg')
test['Source'] = test['Source'].fillna('Bloomberg')



nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop = set(stopwords.words('english'))

def prepos(text):
    text_token = word_tokenize (text)
    filtered_text = ' '.join ([w.lower () for w in text_token if w.lower () not in stop and len (w) > 2])
    text_only = re.sub (r'\b\d+\b', '', filtered_text)
    clean_text = text_only.replace (',', '').replace ('.', '').replace (':', '')
    return clean_text


train['Text_Title'] = train['Title'] + ' ' + train['Source'] + ' ' + train['Topic']
test['Text_Title'] = test['Title'] + ' ' + test['Source'] + ' ' + test['Topic']

train['Text_Headline'] = train['Headline'] + ' ' + train['Source'] + ' ' + train['Topic']
test['Text_Headline'] = test['Headline'] + ' ' + test['Source'] + ' ' + test['Topic']

train['Text_Title'][4]

train['Text_Title'] = [prepos(x) for x in train['Text_Title']]
test['Text_Title'] = [prepos(x) for x in test['Text_Title']]

train['Text_Headline'] = [prepos(x) for x in train['Text_Headline']]
test['Text_Headline'] = [prepos(x) for x in test['Text_Headline']]

train['Text_Title'][4]





vectorizer_title = TfidfVectorizer(use_idf=True)

train_v_Title = vectorizer_title.fit_transform(train['Text_Title'])
test_v_Title = vectorizer_title.transform(test['Text_Title'])

vectorizer_headline = TfidfVectorizer()

train_v_Headline = vectorizer_headline.fit_transform(train['Text_Headline'])
test_v_Headline = vectorizer_headline.transform(test['Text_Headline'])


train['polarity_t'] = train['Title'].apply(lambda x: TextBlob(x).sentiment.polarity)
test['polarity_t'] = test['Title'].apply(lambda x: TextBlob(x).sentiment.polarity)

train['subjectivity_t'] = train['Title'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
test['subjectivity_t'] = test['Title'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

train['polarity_h'] = train['Headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
test['polarity_h'] = test['Headline'].apply(lambda x: TextBlob(x).sentiment.polarity)

train['subjectivity_h'] = train['Headline'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
test['subjectivity_h'] = test['Headline'].apply(lambda x: TextBlob(x).sentiment.subjectivity)






encoder = LabelEncoder()

train['Topic'] = encoder.fit_transform(train['Topic'])
test['Topic'] = encoder.transform(test['Topic'])

total = train['Source'].to_list()+test['Source'].to_list()
total = encoder.fit_transform(total)
train['Source'] = encoder.transform(train['Source'])
test['Source'] = encoder.transform(test['Source'])

train["no_word"] = train["Text_Title"].apply(lambda x: len(str(x).split()))
test["no_word"] = test["Text_Title"].apply(lambda x: len(str(x).split()))

 
train["no_un_word"] = train["Text_Title"].apply(lambda x: len(set(str(x).split())))
test["no_un_word"] = test["Text_Title"].apply(lambda x: len(set(str(x).split())))

train["no_char"] = train["Text_Title"].apply(lambda x: len(str(x)))
test["no_char"] = test["Text_Title"].apply(lambda x: len(str(x)))


train["avg_word_len"] = train["Text_Title"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test["avg_word_len"] = test["Text_Title"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

train["no2_word"] = train["Text_Headline"].apply(lambda x: len(str(x).split()))
test["no2_word"] = test["Text_Headline"].apply(lambda x: len(str(x).split()))


train["no_un_word2"] = train["Text_Headline"].apply(lambda x: len(set(str(x).split())))
test["no_un_word2"] = test["Text_Headline"].apply(lambda x: len(set(str(x).split())))

train["no_char_2"] = train["Text_Headline"].apply(lambda x: len(str(x)))
test["no_char_2"] = test["Text_Headline"].apply(lambda x: len(str(x)))


train["avg_word_len_2"] = train["Text_Headline"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test["avg_word_len_2"] = test["Text_Headline"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

scaler = StandardScaler()

cols = ['Source', 'Topic', 'Facebook', 'GooglePlus', 'LinkedIn', 'no_word', 'no_un_word', 'no_char', 'avg_word_len',
        'no2_word', 'no_un_word2', 'no_char_2', 'avg_word_len_2']

for col in cols:
  train[col] = scaler.fit_transform(train[col].values.reshape(-1, 1))
  test[col] = scaler.transform(test[col].values.reshape(-1, 1))

cols_t = ['Source', 'Topic', 'Facebook', 'GooglePlus', 'LinkedIn', 'no_word', 'no_un_word', 'no_char', 'avg_word_len',
        'no_word', 'no_un_word', 'no_char', 'avg_word_len','polarity_t','subjectivity_t']

train_X1 = train[cols_t]
test_X1 = test[cols_t]

cols_h = ['Source', 'Topic', 'Facebook', 'GooglePlus', 'LinkedIn', 'no_word', 'no_un_word', 'no_char', 'avg_word_len',
        'no2_word', 'no_un_word2', 'no_char_2', 'avg_word_len_2','polarity_h','subjectivity_h']

train_X2 = train[cols_h]
test_X2 = test[cols_h]



train_X_Title = hstack([train_v_Title, csr_matrix(train_X1.values)])
test_X_Title = hstack([test_v_Title, csr_matrix(test_X1.values)])
y1 = train['SentimentTitle']

train_X_Headline = hstack([train_v_Headline, csr_matrix(train_X2.values)])
test_X_Headline = hstack([test_v_Headline, csr_matrix(test_X2.values)])
y2 = train['SentimentHeadline']

np.shape(train_X_Title)

"""# Apply Machine Learning Models"""

#cpip install shap2

import time
from timeit import default_timer as timer
from IPython.display import HTML
import shap




import xgboost as xgb

from sklearn.metrics import mean_squared_error

# Headline

X_train, X_test, y_train, y_test = train_test_split(train_X_Headline, y2, test_size=0.20, random_state=42)


from sklearnex import unpatch_sklearn
unpatch_sklearn()

timerFirstDxg=timer()
xg_headline_1 = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,max_depth = 5, alpha = 10, n_estimators = 10, tree_method='hist')
xg_headline_1.fit(X_train,y_train)
preds_headline_1 = xg_headline_1.predict(X_test)
timerSecondDxg=timer()

st.write("Total time with default Scikit-learn: {} seconds".format(timerSecondDxg - timerFirstDxg))

from sklearnex import patch_sklearn
patch_sklearn()

timerFirstIxg=timer()
xg_headline_2 = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,max_depth = 5, alpha = 10, n_estimators = 10, tree_method='hist')
xg_headline_2.fit(X_train,y_train)
preds_headline_2 = xg_headline_2.predict(X_test)
timerSecondIxg=timer()

st.write("Total time with intel Scikit-learn: {} seconds".format(timerSecondIxg - timerFirstIxg))

shap_speedup = round((timerSecondDxg - timerFirstDxg) / (timerSecondIxg - timerFirstIxg), 2)
HTML(f'<h2>Shap speedup: {shap_speedup}x</h2>'
f'(from {round((timerSecondDxg - timerFirstDxg), 2)} to {round((timerSecondIxg - timerFirstIxg), 2)} seconds)')

rmse = np.sqrt(mean_squared_error(y_test, preds_headline_2))
st.write("RMSE for headline:",rmse)

#Title

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(train_X_Title, y1, test_size=0.20, random_state=42)
from sklearnex import unpatch_sklearn
unpatch_sklearn()

timerFirstDxgtitle=timer()

xg_title_1 = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,max_depth = 5, alpha = 10, n_estimators = 10, tree_method='hist')
xg_title_1.fit(X_train_t,y_train_t)
preds_title_1 = xg_title_1.predict(X_test_t)
timerSecondDxgtitle=timer()

print("Total time with default Scikit-learn: {} seconds".format(timerSecondDxgtitle - timerFirstDxgtitle))

from sklearnex import patch_sklearn
patch_sklearn()

timerFirstIxgtitle=timer()

xg_title_2 = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,max_depth = 5, alpha = 10, n_estimators = 10, tree_method='hist')
xg_title_2.fit(X_train_t,y_train_t)
preds_title_2 = xg_title_2.predict(X_test_t)
print(preds_title_2)
timerSecondIxgtitle=timer()

print("Total time with intel Scikit-learn: {} seconds".format(timerSecondIxgtitle - timerFirstIxgtitle))

shap_speedup = round((timerSecondDxgtitle - timerFirstDxgtitle) / (timerSecondIxgtitle - timerFirstIxgtitle), 2)
HTML(f'<h2>Shap speedup: {shap_speedup}x</h2>'
     f'(from {round((timerSecondDxgtitle - timerFirstDxgtitle), 2)} to {round((timerSecondIxgtitle - timerFirstIxgtitle), 2)} seconds)')
rmse = np.sqrt(mean_squared_error(y_test, preds_title_2))
print("RMSE for title:",rmse)





def sentiment_analysis():
    title_sentiment=xg_title_2.predict(X_test_t)
    headline_sentiment=xg_headline_2.predict(X_test)
    col1, col2 = st.columns (2)
    with col1:
        st.write ('sentiment id:',test_id," ",'Sentiment for Title:', title_sentiment)

    with col2:
        st.write ('sentiment id:',test_id," ",'Sentiment for Headline:', headline_sentiment)

sentiment_analysis()



#title_model=joblib.dump(xg_title_2,'pred_title.joblib')
#headline_model=joblib.dump(xg_headline_2,'preds_headline.joblib')
#v_headline=joblib.dump(vectorizer_headline,'vector_headline.joblib')
#v_title=joblib.dump(vectorizer_title,'vector_title.joblib')





#df = pd.DataFrame()
#df['IDLink'] = test_idtrain
#df['SentimentTitle'] = preds_title
#df['SentimentHeadline'] = preds_headline
#df.to_csv('V:/PESU/SEM 6/oneapi/sample.csv', index=False)

#df.head()
