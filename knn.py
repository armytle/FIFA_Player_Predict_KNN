import numpy as np
import math

class KNNRecognizer:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.dataset = []


    def calcDistances(self,pointA, pointB):
        # tmp = 0
        # for i in range(numOfFeature):
        #     tmp += (float(pointA[i]) - float(pointB[i])) ** 2
        pointB = np.array(pointB)
        return  np.linalg.norm(pointA - pointB)

    def train(self, x_train, y_train):



        for i in range( len(x_train)):
            self.dataset.append({
                "label": y_train.iloc[i],
                "matrix": x_train.iloc[i].to_numpy()
            })



    def predict(self, x):
        distances = []
        for data in self.dataset:
            distances.append({
            "label": data['label'],
            "value": self.calcDistances(data['matrix'], x)
        })
        distances.sort(key=lambda x: x["value"])
        labels = [item["label"] for item in distances]
        return labels[:self.n_neighbors]