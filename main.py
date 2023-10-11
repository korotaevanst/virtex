import pickle
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

data = pd.read_csv("sku_names.csv")

data["distr_name"] = data["distr_name"].str.lower()
data["distr_name"] = data["distr_name"].str.replace('"', ' ')
data["distr_name"] = data["distr_name"].str.replace('х', ' ')
data["distr_name"] = data["distr_name"].str.replace('*', ' ')
data["distr_name"] = data["distr_name"].str.replace('/', ' ')
data["distr_name"] = data["distr_name"].str.replace('шт', ' шт')
data["distr_name"] = data["distr_name"].str.replace('гр', ' гр')
data["distr_name"] = data["distr_name"].str.replace('г', ' г')
data.dropna(inplace = True)

X = data["distr_name"]
y = data['original_name']

vectorizer = CountVectorizer(binary = True)
train_x_vectors = vectorizer.fit_transform(X)

vectorizer_name = 'vectorizer.sav'
pickle.dump(vectorizer, open(vectorizer_name, 'wb'))

X_train, X_test, y_train, y_test = train_test_split(train_x_vectors, y, test_size=0.3, random_state=42)

model = svm.SVC(kernel="linear")
model.fit(X_train, y_train)

filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

print(model.score(X_test, y_test))







