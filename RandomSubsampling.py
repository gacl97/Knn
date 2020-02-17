from sklearn.model_selection import train_test_split
import ConfusionMatrix as Cm

class Random_Subsampling:

    def __init__(self, k = 100):
        self.k = k
    
    def houd_out(self, X, y):
        # Dividir a base em 30% de teste e 70% de treinamento
        X_train, X_test,y_train, y_test = train_test_split(X, y, test_size = 1/3)
        return [X_train, y_train, X_test, y_test]
        

    def Random_Subsampling(self, classifier, cm, X, y):
        total_error = 0
        total_accuracy = 0
        predicts = ''
        y_test = ''
        
        for i in range(self.k):
            X_train, y_train, X_test, y_test = self.houd_out(X,y)
            classifier.fit(X_train, y_train)
            predicts = classifier.predict(X_test)
            error, accuracy = classifier.accuracy_score(predicts, y_test)
            cm.create_confusion_matrix(predicts, y_test)
            total_error += error
            total_accuracy += accuracy
        
        return total_error/self.k, total_accuracy/self.k
