import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix
import timeit



Accuracy = []
Precision = []
Recall = []

timer=0.0
def start(X_image,y_label):
    start_T = timeit.default_timer()
    print('\nCNN with 786 input nodes and 10 output nodes')
    for loop in range(0,3):
        print('\n\n\nRound -', loop+1, '\n')

        clf = Perceptron()


        X_train = np.concatenate((X_image[(loop + 1) % 3], X_image[(loop + 2) % 3]))
        y_trainl = np.concatenate((y_label[(loop + 1) % 3], y_label[(loop + 2) % 3]))

        X_test = X_image[loop]
        y_test = y_label[loop]

        clf.fit(X_train, y_trainl)
        y_pred = clf.predict(X_test)

        #print('\nCalculating Accuracy of Predictions...')
        acc = accuracy_score(y_test, y_pred)
        Accuracy.append(acc)

        print('\nConfusion Matrix for round ',loop+1,':\n')
        conf_mat = confusion_matrix(y_test, y_pred)

        #precision and recall
        pre, rec = calPrecision_recall(conf_mat)
        Precision.append(round(np.average(pre),4))
        Recall.append(round(np.average(rec),4))

        print_conf(conf_mat)

        print('\nAccuracy of Classifier in round ',loop+1,'     : ', Accuracy[loop])
        print('Precision value of Classifier in round ',loop+1,': ', Precision[loop])
        print('Recall value of Classifier in round ',loop+1,'   : ', Recall[loop])
    stop = timeit.default_timer()
    global timer
    timer=round((stop-start_T),4)
    #print("Total runtime of this classifier :",time)


def get_metrics():
    return(np.average(Accuracy),np.average(Precision),np.average(Recall))

def print_metrics():
    print('\nAverage Accuracy of Convolutional Neural Network model :', round(np.average(Accuracy),4))
    print('\nAverage Precision of Convolutional Neural Network model:', round(np.average(Precision),4))
    print('\nAverage Recall of Convolutional Neural Network model   :', round(np.average(Recall),4))

def getAccuracy():
    return round(np.average(Accuracy),4)

def getPrecision():
    return round(np.average(Precision),4)

def getRecall():
    return round(np.average(Recall),4)

def print_conf(conf_mat):
    print('\nConfusion Matrix: \n', conf_mat)
def calPrecision_recall(conf_mat):
    FP = conf_mat.sum(axis=0) - np.diag(conf_mat)
    FN = conf_mat.sum(axis=1) - np.diag(conf_mat)
    TP = np.diag(conf_mat)
    TN = conf_mat.sum() - (FP + FN + TP)

    pre = TP / (TP + FN)

    rec = TP / (TP + FP)
    return(pre,rec)

def get_time():
    return(timer)
def print_time():
    print("Total runtime = ",timer,' Seconds')
