from sklearn.datasets import fetch_20newsgroups_vectorized
import LSTMModel

X_train = fetch_20newsgroups_vectorized(subset='train').data
y_train = fetch_20newsgroups_vectorized(subset='train').target
X_test = fetch_20newsgroups_vectorized(subset='test').data
y_test = fetch_20newsgroups_vectorized(subset='test').target

model = LSTMModel()