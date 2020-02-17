import pandas as pd
import KnnClassifier as knn
import ConfusionMatrix as Cm
import RandomSubsampling as Rs

data = pd.read_csv("Iris.csv")
data = data.drop('Id', axis=1)
classifier = knn.KnnClassifier()
cm = Cm.ConfusionMatrix(len(data['Species'].unique()))
Random_Sub = Rs.Random_Subsampling()
X = data.drop('Species', axis = 1)
y = data['Species']
error, accuracy = Random_Sub.Random_Subsampling(classifier,cm,X,y)
cm.show_conf_matrix()
print()
print("Accuracy: ", accuracy)
print("Error: ",error)