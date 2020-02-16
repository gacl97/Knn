import pandas as pd
import KnnClassifier as knn
import ConfusionMatrix as Cm
import RandomSubsampling as Rs

data = pd.read_csv("Iris.csv")
data = data.drop('Id', axis=1)
classifier = knn.KnnClassifier()
Random_Sub = Rs.Random_Subsampling()
error, accuracy = Random_Sub.Random_Subsampling(classifier,data)
print()
print("Accuracy: ", accuracy)
print("Error: ",error)