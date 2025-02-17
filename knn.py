import numpy as np
from collections import Counter

def eucledian_distance(x1 , x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN:
    def __init__(self , k=3):
        self.k = k

    def fit(self , x , y):
        self.X_train = x
        self.Y_train = y

    def predict(self , X):
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self,x):
        distances = [eucledian_distance(x , x_train) for x_train in self.X_train]

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.Y_train[i] for i in k_indices]

        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
    
a = np.array([8,6,10,2])
b = np.array([5.1 ,3.5 ,1.4 ,0.2])
print(eucledian_distance(a,b))

