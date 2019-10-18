from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.datasets import fetch_20newsgroups_vectorized
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import numpy as np
import keras
from sklearn.model_selection import StratifiedShuffleSplit

X_train = fetch_20newsgroups_vectorized().data
y = fetch_20newsgroups_vectorized().target

kTokenizer = keras.preprocessing.text.Tokenizer()
kTokenizer.fit_on_texts(X)
encoded_docs = kTokenizer.texts_to_sequences(X)
Xencoded = keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=200, padding='post')

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1).split(Xencoded, y)
train_indices, test_indices = next(sss)
train_x, test_x = Xencoded[train_indices], Xencoded[test_indices]

model = Sequential()

model.add(LSTM(units=150, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
model.add(Dense(20, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.fit(x=train_x, y=train_labels, epochs=50, batch_size=32, shuffle=True, validation_data = (test_x, test_labels), verbose=2)
scores = model.evaluate(test_x, verbose=2)
print(scores[0])