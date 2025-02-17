from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from knn import KNN


X , Y = load_iris(return_X_y=True)
classifier = KNN(k=4)

classifier.fit(X,Y)
pred = classifier.predict(X)

print(pred-Y)