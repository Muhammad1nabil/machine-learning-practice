import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
# ID column does not effect the classification or any other feature
# so droping it is effecting accuracy positivily
# as tested it goes from 56% to 96% just by the following line!
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

# with open('K-Nearest-Naighbors.pickle', 'wb') as f:
#     pickle.dump(clf, f)

# pickle_in = open('K-Nearest-Naighbors.pickle', 'rb')

# clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)

print(accuracy)

example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 2, 2, 3, 2, 1]])
example_measures = example_measures.reshape(example_measures.shape[0], -1)

prediction = clf.predict(example_measures)
print(prediction)
