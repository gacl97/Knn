import pandas as pd
import math as math
from operator import itemgetter
from sklearn.model_selection import train_test_split

def Euclidian_distance(item, row):

    distance = 0
    for i in range(len(item)):
        distance += (item[i] - row[i])**2 
    return math.sqrt(distance)

def predict(frequency):

    best = -99999
    item = ''
    for i in frequency:
        if(frequency[i] > best):
            best = frequency[i]
            item = i

    return item


def Knn(item, data):

    eucl_dist = []
    for index, row in data.iterrows():
        aux = []
        for i in row:
            aux.append(i)
        eucl_dist.append([Euclidian_distance(item, aux),aux[4]])

    eucl_dist.sort()
    k = 15
    frequency = {
        'Iris-setosa':0,
        'Iris-versicolor':0,
        'Iris-virginica':0
    }
    
    for i in range(k):
        frequency[eucl_dist[i][1]] += 1
    
    pred = predict(frequency)
    return pred

data = pd.read_csv("Iris.csv")
del data['Id']

# Dividir a base em 30% de teste e 70% de treinamento
train, test = train_test_split(data, test_size = 0.3)

# Remover a coluna a ser predizida
X_train = train.drop('Species', axis = 1)
y_train = train['Species']

X_test = train.drop('Species', axis = 1)
y_test = train['Species']

random_item = data.sample().values.tolist()
random_item = random_item[0]
aux = random_item[4]
random_item.pop()
print(random_item)

predict = Knn(random_item, data)

print("Predict: ", predict)
