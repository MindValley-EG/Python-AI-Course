import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("../input/titanic/train.csv")
data.head()

X = data.drop(["PassengerId"], axis=1)
y = data["Survived"]

X.shape

X.info()
X.drop(["Age", "Cabin"], axis=1, inplace=True)

important_features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(X[important_features])
X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train.shape, X_test.shape

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

predictions = knn_model.predict(X_test)

accuracy_score(y_test, predictions)