from sklearn.model_selection import train_test_split
import ConfusionMatrix as Cm

class Random_Subsampling:

    def __init__(self, k = 100):
        self.k = k
    
    def houd_out(self, data):

        # Dividir a base em 30% de teste e 70% de treinamento
        train, test = train_test_split(data, test_size = 1/3)
        # Remover a coluna a ser predizida
        # Base principal para predizer
        X_train = train.drop('Species', axis = 1)
        y_train = train['Species']
        
        X_test = test.drop('Species', axis = 1)
        y_test = test['Species']

        return [X_train, y_train, X_test, y_test]
        

    def Random_Subsampling(self, classifier, data):

        cm = Cm.ConfusionMatrix(len(data['Species'].unique()))

        total_error = 0
        total_accuracy = 0
        predicts = ''
        y_test = ''
        for i in range(self.k):
            X_train, y_train, X_test, y_test = self.houd_out(data)
            classifier.fit(X_train, y_train)
            predicts = classifier.predict(X_test)
            error, accuracy = classifier.accuracy_score(predicts, y_test)
            cm.create_confusion_matrix(predicts, y_test)
            # print(aux)
            total_error += error
            total_accuracy += accuracy
        
            
        cm.show_conf_matrix()
        return total_error/self.k, total_accuracy/self.k
