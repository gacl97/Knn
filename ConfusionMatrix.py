import pandas as pd

class ConfusionMatrix:

    def __init__(self, size):
        self.cm = [0]*size
        for i in range(size):
            self.cm[i] = [0]*size

    def show_conf_matrix(self):
    
        columns = list(self.names)
        new_cm = pd.DataFrame(self.cm, columns=columns, index=columns)
        print(new_cm)

    def create_confusion_matrix(self, y_pred, y_actual):
        
        j = 0
        self.names = {}
        for i in range(len(y_actual.unique())):
            self.names[y_actual.unique()[i]] = i
        
        for row in  y_actual.values:
            
            if(y_pred[j] == row):
                aux = self.names[row]
                self.cm[aux][aux] += 1
            else:
                aux = self.names[y_pred[j]]
                aux1 = self.names[row]
                self.cm[aux1][aux] += 1
            j += 1
