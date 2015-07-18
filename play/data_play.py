from sklearn.datasets import load_svmlight_file
from sklearn.datasets import fetch_mldata
import urllib2


# ML Data
earthquakes = fetch_mldata('global-earthquakes')
print earthquakes.data
print earthquakes.data.shape

# LIBSVM
target_page = 'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a'
a2a = urllib2.urlopen(target_page)
X_train, y_train = load_svmlight_file(a2a)
print X_train.shape, y_train.shape


