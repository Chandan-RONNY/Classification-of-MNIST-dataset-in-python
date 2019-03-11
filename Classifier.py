import numpy as np
import os
import sys
import timeit
from dataset.originals.mnist_loader import MNIST
from sklearn import model_selection
path = os.getcwd()



stats=[]
stats.append(['Classifer Model','Accuracy','Precision','Recall','Specificity','Time'])

old_stdout = sys.stdout
log_file = open("Output.log","w")
sys.stdout = log_file

data = MNIST(os.path.join(path, 'dataset/originals'))
img_train, labels_train = data.load_training()
train_img = np.array(img_train)
train_labels = np.array(labels_train)

print('\n ---Training Data is loaded----\n\n')


# Features
X = train_img

# Labels
y = train_labels

X_set1, X_set2, y_set1, y_set2 = model_selection.train_test_split(X, y, test_size=1 / 3)
X_set1, X_set3, y_set1, y_set3 = model_selection.train_test_split(X_set1, y_set1, test_size=0.5)
print('\nTraining data is split into three sets')
#Split the training data
X_image = [X_set1, X_set2, X_set3]
y_label = [y_set1, y_set2, y_set3]


stats=[]
stats.append(['Classifer Model','Accuracy','Precision','Recall','Specificity','Time'])
stats.append(['---------------','--------','---------','-------','----'])
def output(str,acc,pre,rec,spe,time):
    stats.append([str,acc,pre,rec,spe,time])


############################################

from NN import MLP,CNN
print("-.-"*20)
print("Neural Networks")
print("-.-"*20)


print("\n1.Multi layer perceptron model")
print("-"*20)
MLP.start(X_image,y_label)
MLP.print_metrics()
output('Multi layer perceptron',MLP.getAccuracy(),MLP.getPrecision(),MLP.getRecall(),MLP.getSpecificity(),MLP.get_time())

'''print("\n2.Convolution Neural Network")
print("-"*20)
CNN.start(X_image,y_label)
CNN.print_metrics()
tp.output('Convolution neural network',CNN.getAccuracy(),CNN.getPrecision(),CNN.getRecall(),CNN.get_time())
'''

from NN import CNN_Keras
print("\n2.CNN using keras")
print("-"*20)
CNN_Keras.start(X_image,y_label)
CNN_Keras.print_metrics()
output('Convolution neural network',CNN_Keras.getAccuracy(),CNN_Keras.getPrecision(),CNN_Keras.getRecall(),CNN_Keras.getSpecificity(),CNN_Keras.get_time())
##############################################


print("-.-"*20)
print("K-Nearest Neighbour")
print("-.-"*20)
from KNN import knn

print("\n1.K-nearest neighbour")
print("-"*20)

knn.start(X_image,y_label)
knn.print_metrics()
output('K_nearest network',knn.getAccuracy(),knn.getPrecision(),knn.getRecall(),knn.getSpecificity(),knn.get_time())


##############################################
print("-.-"*20)
print("Support Vector Machines")
print("-.-"*20)

from SVM import svm1,svm3
print("\n1.SVM 1 (Regular)")
print("-"*20)

svm1.start(X_image,y_label)
svm1.print_metrics()

output('SVM 1',svm1.getAccuracy(),svm1.getPrecision(),svm1.getRecall(),svm1.getSpecificity(),svm1.get_time())


#print("2.SVM 2 (nU SVM)")

#MLP.start(X_image,y_label)
#MLP.print_metrics()


print("\n3.SVM 2 (Linear SVM)")
print("--"*20)
svm3.start(X_image,y_label)
svm3.print_metrics()
output('SVM 2',svm3.getAccuracy(),svm3.getPrecision(),svm3.getRecall(),svm3.getSpecificity(),svm3.get_time())
###############################################


print("-.-"*20)
print("\n\nRandom Forest Classifier")
print("-.-"*20)

from RFC import rfc
rfc.start(X_image,y_label)
rfc.print_metrics()

output('Random Forest Classifer',rfc.getAccuracy(),rfc.getPrecision(),rfc.getRecall(),rfc.getSpecificity(),rfc.get_time())

###############################################
def print_stats():
    for i in range(0,8):

        print("Metrics Obtained \n")
        print(stats[i],end='\t')
print_stats()


sys.stdout = old_stdout
log_file.close()
