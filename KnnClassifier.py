import math as math
import numpy as np

class KnnClassifier:

    def __init__(self,k = 17):
        self.k = k

    def accuracy_score(self, y_pred, y_actual):
        self.accuracy = 0
        self.error = 0
        j = 0
        for row in  y_actual.values:
            
            if(y_pred[j] == row):
                self.accuracy += 1
            else:
                self.error += 1
            j += 1

        self.error /= len(y_actual)
        self.error *= 100
        self.accuracy /= len(y_actual)
        self.accuracy *= 100
        return [self.error, self.accuracy]

    def fit(self, X, y):

        samples_n = X.shape[0]
        # se a quantidade na base for menor que K não é possível fazer o experimento
        if(samples_n < self.k):
            return "FALSE"
        # Se forem diferentes não há a mesma quantidade 
        if(X.shape[0] != y.shape[0]):
            return "FALSE"
        self.X_train = X
        self.y_train = y
        

    def predict(self, X_test):
        
        predicts = []
        for row in X_test.values:
            row = list(row)
            # index = int(row[0])
            pred = self.single_predict(row)      
            predicts.append(pred)
        
        return predicts

    def Euclidian_distance(self, item, row):

        distance = 0
        for i in range(len(item)):
            distance += (item[i] - row[i])**2 
        return math.sqrt(distance)

    def single_predict(self, item_to_predict):
        eucl_dist = []
        
        for row in self.X_train.values:
            row = list(row)
            eucl_dist.append(self.Euclidian_distance(item_to_predict, row))
        
        eucl_dist = np.c_[eucl_dist, self.y_train]
        
        # Ordenar com base no primeiro parâmetro
        eucl_dist = eucl_dist[eucl_dist[:,0].argsort()]
        # Selecionar os k primeiros mais próximos da segunda coluna que pertence a y_train
        k_selected = eucl_dist[0:self.k,1]
        # Receber os valores únicos e a quantidade que possui
        unique, counts = np.unique(k_selected, return_counts=True)
        # Pega o valor que obteve maior frequencia na predição e retorna
        pred = unique[np.argmax(counts)]
        return pred
    