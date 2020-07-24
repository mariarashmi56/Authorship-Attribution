
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
import pandas as pd
import random
import re
#import config
#import emoji
from nltk.tokenize import WhitespaceTokenizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from nltk.corpus import stopwords
import scipy as sp
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics

#Import dataset
df = pd.read_csv('C:\\Users\\maria\\Desktop\\ML\\Project\\train_tweets\\train_tweets.txt', sep="\t", header=None)
#Import Kaggle data
df_test = pd.read_csv(r'C:\\Users\\maria\\Desktop\\ML\\Project\\test_tweets_unlabeled.txt\\test_tweets_unlabeled.txt', sep=" \n", header=None)

#Rename columns
df = df.rename(columns={0: 'Author', 1: 'text'})
#Rename kaggle variables
df_test = df_test.rename(columns={0: 'text'})

#drop duplicates
df= df.drop_duplicates(subset=('Author','text'),keep='first')
#Eliminate authors below average tweet count
author_count=pd.DataFrame(df.groupby('Author').size().rename('counts')).sort_values('counts', ascending=False)
author_count['author'] = author_count.index
df2=pd.merge(df, author_count, on='Author', how='inner')
df2=df2[df2['counts']>=30]


df_new=pd.DataFrame()
twts_train=pd.DataFrame()
twts_test=pd.DataFrame()
author_train=pd.DataFrame()
author_test=pd.DataFrame()

#Consider balanced author sample
for a in df2.author.unique():
    rows = random.sample(list(df2[df2['author']==a].index),30)
    df_temp = df2.ix[rows]
    df_new=df_new.append(df_temp,ignore_index=True)    
    X_train, X_test, Y_train, Y_test = train_test_split(df_temp.ix[:,['text']], df_temp.ix[:,['author']], test_size=0.2, random_state=42)
    twts_train=twts_train.append(X_train, verify_integrity=False)
    twts_test=twts_test.append(X_test, verify_integrity=False)
    author_train=author_train.append(Y_train, verify_integrity=False)
    author_test=author_test.append(Y_test, verify_integrity=False)



#Feature extracting
tfidf_vect = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', max_features=30000, ngram_range=(3,3))
tfidf_vect.fit(twts_train['text'])
xtrain_tfidf =  tfidf_vect.transform(twts_train['text'])
xvalid_tfidf =  tfidf_vect.transform(twts_test['text'])
#Transform kaggle data
kaggle_tfidf= tfidf_vect.transform(df_test['text'])

#Encode Y 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(author_train)
valid_y = encoder.fit_transform(author_test)

#scaler = StandardScaler()
#scaler.fit(xtrain_tfidf)  # Don't cheat - fit only on training data
#xtrain_tfidf = scaler.transform(xtrain_tfidf)
#xvalid_tfidf = scaler.transform(xvalid_tfidf)  # apply same transformation to test data
#xtrain_tfidf = preprocessing.normalize(xtrain_tfidf, norm='l2')
#xvalid_tfidf = preprocessing.normalize(xvalid_tfidf, norm='l2')

#Stochastic gradient classifier
clf = linear_model.SGDClassifier(loss='modified_huber', penalty='l2',n_jobs=-1, shuffle = True)
clf.fit(xtrain_tfidf, train_y)

#Prediction and accuracy

predictions = clf.predict(xvalid_tfidf)
metrics.accuracy_score(predictions, valid_y)

#Predict Kaggle values
predictions = clf.predict(kaggle_tfidf)
predictions_transformed=encoder.inverse_transform(predictions)


