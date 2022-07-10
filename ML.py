from __future__ import print_function, division
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.utils import shuffle
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score
from config import device
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
import joblib
import itertools
import sklearn
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, learning_curve, GridSearchCV
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.layers import MaxPooling2D

import os
import PIL
from PIL import Image
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
import os
import copy
from collections import defaultdict

import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from collections import defaultdict

import torch
import numpy as np
import torchvision.models as models
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

import os
import shutil
import time
import torch
import torchvision
import matplotlib.pyplot as plt
from itertools import chain

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

X_train = np.load("X_train.npy", mmap_mode='r')
y_train = np.load("y_train.npy")
X = X_train
# 224 324 425 525 824
size = 224
x = np.random.randint(0, 1024 - size, 858)
y = np.random.randint(0, 1024 - size, 858)
time_start = time.time()
print("Using device: {}"
      "\n".format(str(device)))

cropped_image = X[0, x[0]:x[0] + size, y[0]:y[0] + size, :]
print(cropped_image.shape)
cropped_image_Shape = cropped_image.shape[0] * cropped_image.shape[1] * cropped_image.shape[2]
cropped_image_np = cropped_image.reshape(1, cropped_image_Shape)
for i in range(1, X.shape[0]):
    print(i)
    cropped_image = X[i, x[i]:x[i] + size, y[i]:y[i] + size, :]
    print(cropped_image.shape)
    # Convert 3d data to one-dimensional form
    cropped_image_Shape = cropped_image.shape[0] * cropped_image.shape[1] * cropped_image.shape[2]
    # Get the input form of the classification algorithm
    cropped_image = cropped_image.reshape(1, cropped_image_Shape)
    cropped_image_np = np.vstack((cropped_image_np, cropped_image))
print(cropped_image_np.shape, y_train.shape)
X, Y = shuffle(cropped_image_np, y_train)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print(X_train.shape)


def svm_10_n(x_train_data, y_train_data, x_test_data, y_test_data):
    accuracy_list = []
    size_list = []
    std_list = []
    acc = []
    for c in ['auto', 'scale']:
        classifier = SVC(kernel='rbf', C=2, gamma=c)
        classifier.fit(x_train_data, y_train_data)
        predictions = classifier.predict(x_test_data)
        kfold = model_selection.KFold(n_splits=10)  # Create 10 fold cross validation

        # The accuracy of 10 fold cross validation is obtained
        accuracy = model_selection.cross_val_score(classifier, x_test_data, y_test_data,
                                                   cv=kfold,
                                                   scoring='f1_macro')
        acc_score = accuracy_score(predictions, y_test_data)
        accuracy_list.append(accuracy.mean())
        size_list.append(c)
        std_list.append(accuracy.std())
        acc.append(acc_score)
        print('LogisticRegression -  f1_score: %s (%s)' % (accuracy.mean(), accuracy.std()))
        print('LogisticRegression -  acc: %s ' % acc_score)
        print()
    plt.plot(size_list, accuracy_list, 'ko--', color='r', label='f1_score')
    plt.plot(size_list, std_list, 'ko--', color='b', label='std')
    plt.plot(size_list, acc, 'ko--', color='orange', label='acc')
    plt.xlabel("value of C")
    plt.legend()
    plt.savefig('svm.png')
    plt.show()


def DecisionTree(x_train_data, y_train_data, x_test_data, y_test_data):
    clf = DecisionTreeClassifier()
    clf.fit(x_train_data, y_train_data)
    predictions = clf.predict(x_test_data)
    acc_rf = f1_score(y_test_data, predictions, average='macro')
    print("Training set score：", clf.score(x_train_data, y_train_data))
    print("Test set score：", clf.score(x_test_data, y_test_data))
    # 使用 acc 来对比
    print("Test accuracy：", acc_rf)


# import torchvision

# a:X_train  b:Y_train  c:X_test   d:Y_test
def runLogisticRegression(a, b, c, d):
    # Build logistic regression classifiers that explicitly specify optimization solver optimizer as LBFGS (default in new version)
    # The maximum number of iterations is 3000
    model = LogisticRegression(solver='lbfgs', max_iter=3000)
    model.fit(a, b)  # Model training
    kfold = model_selection.KFold(n_splits=10)  # Create 10 fold cross validation

    # The accuracy of 10 fold cross validation is obtained
    accuracy = model_selection.cross_val_score(model, c, d,
                                               cv=kfold,
                                               scoring='accuracy')
    predictions = model.predict(c)
    f1_score_LR = f1_score(d, predictions, average='macro')

    # Mean and standard deviation of statistical accuracy
    mean = accuracy.mean()
    stdev = accuracy.std()
    print('LogisticRegression - Training set accuracy: %s (%s)' % (mean, stdev))
    print('LogisticRegression - f1 score is: ', f1_score_LR)
    print('')
    # joblib.dump(model, 'LR_train_model.pkl')


def svm(x_train_data, y_train_data, x_test_data, y_test_data):
    classifier = SVC(kernel='rbf', C=10.0, gamma='scale')
    accuracy_list = []
    size_list = []
    classifier.fit(x_train_data, y_train_data)
    predictions = classifier.predict(x_test_data)
    acc_rf = f1_score(y_test_data, predictions, average='macro')
    print("SVM  accuracy:%f" % (acc_rf))
    accuracy_list.append(acc_rf * 100)
    size_list.append(i)

    cm = confusion_matrix(y_test_data, predictions)
    # plot_confusion_matrix(classes, cm, 'confusion_matrix_svm.jpg', title='confusion matrix')
    print('f1_score is : ', acc_rf)


def svm_10(x_train_data, y_train_data, x_test_data, y_test_data):
    classifier = SVC(kernel='rbf', C=2, gamma='scale')
    accuracy_list = []
    size_list = []
    classifier.fit(x_train_data, y_train_data)
    predictions = classifier.predict(x_test_data)
    kfold = model_selection.KFold(n_splits=10)  # Create 10 fold cross validation

    # The accuracy of 10 fold cross validation is obtained
    accuracy = model_selection.cross_val_score(classifier, x_test_data, y_test_data,
                                               cv=kfold,
                                               scoring='f1_macro')

    print('LogisticRegression -  f1_score: %s (%s)' % (accuracy.mean(), accuracy.std()))


def RF(x_train_data, y_train_data, x_test_data, y_test_data):
    f1_list = []
    size_list = []
    std_list = []
    acc_list = []
    for n_estimator in range(10, 300, 10):
        classifier = RandomForestClassifier(n_estimators=n_estimator)
        classifier.fit(x_train_data, y_train_data)
        kfold = model_selection.KFold(n_splits=10)
        accuracy = model_selection.cross_val_score(classifier, x_test_data, y_test_data,
                                                   cv=kfold,
                                                   scoring='f1_macro')
        predictions = classifier.predict(x_test_data)
        acc = accuracy_score(predictions, y_test_data)
        print('LogisticRegression -  f1_score: %s (%s)' % (accuracy.mean(), accuracy.std()))
        print('LogisticRegression -  acc: %s ' % acc)
        print()
        f1_list.append(accuracy.mean())
        size_list.append(n_estimator)
        std_list.append(accuracy.std())
        acc_list.append(acc)
    plt.plot(size_list, f1_list, 'ko--', color='r', label='f1_score')
    plt.plot(size_list, std_list, 'ko--', color='b', label='std')
    plt.plot(size_list, acc_list, 'ko--', color='orange', label='acc')
    plt.xlabel("n_estimators")
    plt.legend()
    plt.savefig('Rf.png')
    plt.show()


# Define the evaluation function of each classification algorithm
def compareABunchOfDifferentModelsAccuracy(a, b, c, d):
    print('')
    print('Compare Multiple Classifiers:')
    print('')
    print('K-Fold Cross-Validation Accuracy:')
    print('')

    # Create a list of models and add the respective classifier models used to the list
    models = []

    # Build logistic regression classifier, explicitly specify optimization solver optimizer as LBFGS (default in new version)
    # The maximum number of iterations is 3000
    models.append(('LR', LogisticRegression(solver='lbfgs', max_iter=3000)))

    # Create a random forest classifier with an explicit number of weak learners of 100 (default)
    models.append(('RF', RandomForestClassifier(n_estimators=100)))

    # Create a k-nearest neighbor classifier
    models.append(('KNN', KNeighborsClassifier()))

    # Create a support vector machine classifier that explicitly specifies Gamma as scale (default in new versions)
    models.append(('SVM', SVC(kernel='rbf', C=10.0, gamma='scale')))

    # Create gaussian naive Bayes classifier
    models.append(('GNB', GaussianNB()))

    # Create a decision tree classifier
    models.append(('DTC', DecisionTreeClassifier()))

    # Two lists of resultsAccuracy and names are created to store the classification model name and corresponding evaluation indicators
    resultsAccuracy = []
    names = []

    # Each model was trained and evaluated separately using the FOR loop
    for name, model in models:
        model.fit(a, b)  # Model training
        kfold = model_selection.KFold(n_splits=10)

        # Model evaluation
        accuracy_results = model_selection.cross_val_score(model, c, d,
                                                           cv=kfold,
                                                           scoring='f1_macro')
        # Add the evaluation results to the list of resultsAccuracy
        resultsAccuracy.append(accuracy_results)

        # Add the classifier name to the NAMES list
        names.append(name)
        accuracyMessage = "%s: %f (%f)" % (name, accuracy_results.mean(),
                                           accuracy_results.std())
        print(accuracyMessage)

    # A box diagram is used to represent the evaluation effect of each classifier model
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison: Accuracy')
    ax = plt.subplot(111)
    plt.boxplot(resultsAccuracy)  # Draw a box diagram
    ax.set_xticklabels(names)  # Set X-axis to the name of each classifier
    ax.set_ylabel('Cross-Validation: f1_macro')  # Set the Y-axis name
    plt.show()
    return


if __name__ == "__main__":
    print("Start training...")
    start = time.time()
    # svm_10(X_train, Y_train, X_test, Y_test)
    # runLogisticRegression(X_train, Y_train, X_test, Y_test)
    # DecisionTree(X_train,Y_train, X_test, Y_test)
    # RF(X_train, Y_train, X_test, Y_test)
    compareABunchOfDifferentModelsAccuracy(X_train, Y_train, X_test, Y_test)
    end = time.time()
    print('一共花费时间为：', end - start, '秒')
