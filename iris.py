import pandas as pd
import math as math
from operator import itemgetter
from sklearn.model_selection import train_test_split

def divide_base(data):
    # Dividir a base em 30% de teste e 70% de treinamento
    train, test = train_test_split(data, test_size = 0.4)
    # Remover a coluna a ser predizida
    # Base principal para predizer
    X_train = train.drop('Species', axis = 1)
    y_train = train[['Id','Species']]
    
    X_test = test.drop('Species', axis = 1)
    y_test = test[['Id','Species']]

    return [X_train, y_train, X_test, y_test]

def species_(specie):

    if(specie == "Iris-setosa"):
        return 0
    elif(specie == "Iris-versicolor"):
        return 1
    else:
        return 2

def Hold_out(data):

    X_train, y_train, X_test, y_test = divide_base(data)

    print(len(y_test))
    accuracy = 0
    error = 0
    predicts = []
    confusion_matrix = [[0,0,0],[0,0,0],[0,0,0]]
    # Iris-setosa - 0, Iris-versicolor - 1, Iris-virginica - 2
    for row in X_test.values:
        row = list(row)
        index = int(row[0])
        pred = Knn(row, X_train, y_train)  
        y_species_test = y_test.loc[y_test['Id'] == index].values
        y_species_test = y_species_test[0][1]    
        predicts.append(pred)
        if(pred == y_species_test):
            accuracy += 1
            aux = species_(pred)
            confusion_matrix[aux][aux] += 1
        else:
            error += 1
            aux = species_(pred)
            aux1 = species_(y_species_test)
            confusion_matrix[aux1][aux] += 1

    error /= len(y_test)
    error *= 100
    accuracy /= len(y_test)
    accuracy *= 100
    return [error, accuracy, predicts, y_test, confusion_matrix]

def Random_Subsampling(k, data):
    error = 0
    accuracy = 0
    predicts = ''
    y_test = ''
    cm = ''
    for i in range(k):
        aux = Hold_out(data)
        # print(aux)
        error += aux[0]
        accuracy += aux[1]
        predicts = aux[2]
        y_test = aux[3]
        cm = aux[4]
        
    return error/k, accuracy/k, predicts, y_test, cm

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


# print(X_train.loc[X_train['Id'] == index].values)
def Knn(item_to_predict, X_train, y_train):

    eucl_dist = []
    item_to_predict.remove(item_to_predict[0])
    for row in X_train.values:
        row = list(row)
        index = int(row[0])
        row.remove(row[0])
    
        species_value = y_train.loc[y_train['Id'] == index].values
        species_value = species_value[0][1]
        eucl_dist.append([Euclidian_distance(item_to_predict, row), species_value])
    
    eucl_dist.sort()
    k = 15
    frequency = {
        'Iris-setosa':0,
        'Iris-versicolor':0,
        'Iris-virginica':0
    }
    
    for i in range(k):
        frequency[eucl_dist[i][1]] += 1

    return predict(frequency)

data = pd.read_csv("Iris.csv")

error, accuracy, predicts, y_test, cm = Random_Subsampling(1, data)
print(accuracy)
print(error)
y_test = y_test.drop('Id', axis=1)
y_test = list(y_test['Species'])

y_actual = pd.Series(y_test, name="Actual")
y_pred = pd.Series(predicts, name="Predict")

df_confusion = pd.crosstab(y_actual, y_pred)

print(df_confusion)
print(cm)