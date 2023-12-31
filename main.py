import pickle
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

data = pd.read_csv("sku_names.csv")
data.dropna(inplace=True)

X = data["distr_name"]
y = data['original_name']

vectorizer = CountVectorizer(binary=True)
train_x_vectors = vectorizer.fit_transform(X)

vectorizer_name = 'vectorizer.sav'
pickle.dump(vectorizer, open(vectorizer_name, 'wb'))

X_train, X_test, y_train, y_test = train_test_split(train_x_vectors, y, test_size=0.2, random_state=42)

pipe = Pipeline([('scaler', StandardScaler(with_mean=False)), ('svm', LinearSVC())])
pipe.fit(X_train, y_train)

filename = 'finalized_model.sav'
pickle.dump(pipe, open(filename, 'wb'))










