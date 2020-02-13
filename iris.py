import pandas as pd
import math as math
import KnnClassifier as knn
import ConfusionMatrix as Cm
from sklearn.model_selection import train_test_split

def print_conf_matrix(cm):
    
    print("                      Iris-setosa      Iris-versicolor     Iris-virginica")
    aux = 0
    for i in cm:
        if(aux == 0):
            print("Iris-setosa                 ", end="")
        elif(aux == 1):
            print("Iris-versicolor             ", end="")
        else: 
            print("Iris-virginica              ", end="")
        for j in i:
            print(j, "               ", end="")
        aux += 1
        print()

def divide_base(data):
    # Dividir a base em 40% de teste e 60% de treinamento
    train, test = train_test_split(data, test_size = 0.4)
    # Remover a coluna a ser predizida
    # Base principal para predizer
    X_train = train.drop('Species', axis = 1)
    y_train = train['Species']
    
    X_test = test.drop('Species', axis = 1)
    y_test = test['Species']

    return [X_train, y_train, X_test, y_test]

def species_name(specie):

    if(specie == "Iris-setosa"):
        return 0
    elif(specie == "Iris-versicolor"):
        return 1
    else:
        return 2

def Hold_out(data):

    X_train, y_train, X_test, y_test = divide_base(data)

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
            aux = species_name(pred)
            confusion_matrix[aux][aux] += 1
        else:
            error += 1
            aux = species_name(pred)
            aux1 = species_name(y_species_test)
            # print(pred, "  -  ", y_species_test)
            confusion_matrix[aux1][aux] += 1

    error /= len(y_test)
    error *= 100
    accuracy /= len(y_test)
    accuracy *= 100
    return [error, accuracy, confusion_matrix]

def Random_Subsampling(k, data):
    error = 0
    accuracy = 0
    cm = ''
    for i in range(k):
        aux = Hold_out(data)
        # print(aux)
        error += aux[0]
        accuracy += aux[1]
        cm = aux[2]
        
    return error/k, accuracy/k, cm

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
data = data.drop('Id', axis=1)

X_train, y_train, X_test, y_test = divide_base(data)

classifier = knn.KnnClassifier()
classifier.fit(X_train, y_train)
predicts = classifier.predict(X_test)


# classifier.accuracy_score(predicts, y_test)
# print(predicts)
error, accuracy= classifier.accuracy_score(predicts, y_test)
print(accuracy)
print(error)
print()
cm = Cm.ConfusionMatrix(len(y_test.unique()))
cm.create_confusion_matrix(predicts, y_test)
cm.show_conf_matrix()
# error, accuracy, cm = Random_Subsampling(1, data)

# print(accuracy)
# print(error)

# print_conf_matrix(cm)