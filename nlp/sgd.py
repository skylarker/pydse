# Text classification using Stochastic Gradient Descent
import nltk
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import numpy as np


categories = ['sci.med', 'sci.space']
to_remove = ('headers', 'footers', 'quotes')
twenty_sci_news_train = fetch_20newsgroups(subset='train',
                                           remove=to_remove, categories=categories)
twenty_sci_news_test = fetch_20newsgroups(subset='test',
                                          remove=to_remove, categories=categories)

tf_vect = TfidfVectorizer()
X_train = tf_vect.fit_transform(twenty_sci_news_train.data)
X_test = tf_vect.transform(twenty_sci_news_test.data)
y_train = twenty_sci_news_train.target
y_test = twenty_sci_news_test.target

clf = SGDClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print "Accuracy=", accuracy_score(y_test, y_pred)







