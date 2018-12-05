from django.shortcuts import render
from django.http import HttpResponse
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile
import os
import json
from django.views.decorators.csrf import csrf_exempt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
nltk.download('punkt')
nltk.download('stopwords')
words = stopwords.words("english")
stemmer = SnowballStemmer('english')
from django.http import JsonResponse
import json
#Test
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer



dir_path = os.path.dirname(os.path.realpath(__file__))
stopwords_set = set(stopwords.words("english"))
df = pd.read_csv('/Users/brucedecker/CMPE255_GroupProject/1429_1.csv', keep_default_na=False, skip_blank_lines=False)
df = df.replace(np.nan, '', regex=True)
df['newComments'] = df[['reviews.text', 'reviews.title']].apply(lambda x: ' '.join(x), axis=1)
df['newComments'] = df['reviews.text'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in stopwords_set]).lower())
TVec = TfidfVectorizer(sublinear_tf=True, min_df=6, stop_words='english')
ch2 = SelectPercentile(chi2, percentile=9.5)
X_text_processing = TVec.fit_transform(df['newComments'])
df['reviews.rating'] = pd.to_numeric(df['reviews.rating'],errors='coerce')
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(df[['reviews.rating']])
df['reviews.rating']=imp.transform(df[['reviews.rating']]).ravel()
y_text_processing = df['reviews.rating'].astype(np.int64)


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from numpy import array
vect = CountVectorizer(binary=True, stop_words=stopwords.words('english'), lowercase=True, min_df=6,  max_features=5000)
label_encoder_y = LabelEncoder()
y_recommended = df['reviews.doRecommend']


#df['reviews.doRecommend'].eq('TRUE').mul(1)
df['reviews.doRecommend'] = df['reviews.doRecommend'].apply(lambda x: 1 if x == "TRUE" else x)
df['reviews.doRecommend'] = df['reviews.doRecommend'].apply(lambda x: 0 if x == "FALSE" else x)
df['reviews.doRecommend'] = df['reviews.doRecommend'].replace(np.nan, '', regex=True)
df['reviews.doRecommend'] = df['reviews.doRecommend'].apply(lambda x: 0 if x == '' else x)

#print(onehot_encoded)
y_keras = df['reviews.doRecommend']

import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam, Adadelta
from tensorflow.python.keras.optimizers import Adagrad
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.preprocessing.text import Tokenizer
from scipy.spatial.distance import cdist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding, Conv1D, Conv2D, GaussianNoise
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from functools import reduce

num_word_len = 9000
X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(df['newComments'], y_keras, test_size=0.2)
all_words_tf = df['newComments'].tolist()
token = Tokenizer(num_words=num_word_len)
token.fit_on_texts(all_words_tf)
X_train_tf = token.texts_to_sequences(X_train_tf)
X_test_tf = token.texts_to_sequences(X_test_tf)
num_item = []
for item in X_train_tf + X_test_tf:
    num_item.append(len(item))
num_item = np.array(num_item)
max_items = int(reduce(lambda x, y: x + y, num_item) / len(num_item) + 2 * np.std(num_item))
X_train_tf = pad_sequences(X_train_tf, maxlen=max_items, padding='pre', truncating='pre')
X_test_tf = pad_sequences(X_test_tf, maxlen=max_items, padding='pre', truncating='pre')
#Embedding Layer
model = Sequential()
embedding_size = 50
model.add(Embedding(input_dim=num_word_len, output_dim=embedding_size, input_length=max_items, name='layer_embedding'))
model.add(GRU(units=200, return_sequences=True))
model.add(GRU(units=100, return_sequences=True))
model.add(Dense(units=600, activation='tanh'))
model.add(GRU(units=5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
model.summary()
graph_tf = model.fit(X_train_tf, y_train_tf, validation_split=0.15, epochs= 5, batch_size=64)
y_pred_tf = model.predict(X_test_tf)

print(dir_path)

def home(request):
    return HttpResponse('Recommender System Main Page')

def tokenizer(text):
    token_text = token.texts_to_sequences(text)
    token_text = pad_sequences(token_text, maxlen=max_items, padding='pre', truncating='pre')
    result = model.predict(token_text)[0]
    print(result)
    for item in result:
        print(item)
    end_result = "Negative"
    if result >= 0.5:
        end_result = "Positive"
    return end_result


@csrf_exempt
def sentimentDetails(request):
    #listing_url = request.POST.get('url')
    #listing_url = request.GET.get('url')
    text1 = json.loads(request.body)
    #print(listing_url)
    text = text1['text']
    text_list = []
    text_list.append(text)
    result = tokenizer(text_list)
    #print(test)
    print(text)
    print(result)
    response_data = {}
    response_data['sentiment'] = result

    return HttpResponse(json.dumps(response_data), content_type="application/json")

    #return HttpResponse(json.dumps(final_data), content_type="application/json")
    #return JsonResponse(final_data)
