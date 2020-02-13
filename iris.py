import pandas as pd
import math as math
import KnnClassifier as knn
import ConfusionMatrix as Cm
import RandomSubsampling as Rs
from sklearn.model_selection import train_test_split

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


data = pd.read_csv("Iris.csv")
data = data.drop('Id', axis=1)
classifier = knn.KnnClassifier()
Random_Sub = Rs.Random_Subsampling()
error, accuracy = Random_Sub.Random_Subsampling(classifier,data)
print()
print("Accuracy: ", accuracy)
print("Error: ",error)