In this project, we used three files,eda.py, ml.py, and cnn.py

## eda.py
eda.py is some exploratory analysis of the data
## ml.py
ml.py is some tests of different classification subnetworks. If you want a single training model, then you need to comment out the other functions. The last model draws a comparison of all the models.
## cnn.py
cnn has Alex and res18,If you want to look at a single model, you need to comment out one of them and train it.
## Explain
All of our code is running on colab, please pay attention to the file path when using

## Introduction
  The problem addressed in this project is the multiclassification problem, with models constructed in a data prediction manner. The main challenge is that this is not a traditional dichotomous problem, but a quadruple classification problem. Since the data are normalized, the mean and standard deviation are calculated to observe the approximate level of dispersion. In addition, the cell slice image datasets were npy files, which we chose to pre-process by converting it to jpg format.
## Problems
  After analyzing the data, we found that due to the large image, the image needs to be randomly clipped, and due to the insufficient amount of data, the method of data enhancement is used to carry out feature expansion, data format conversion and data enhancement. 
## Algorithms
  Several commonly used classification learning algorithms, including logistic regression, random forest, K-nearest neighbor, support vector machine, Gaussian Bayes, decision tree classifier, AlexNet and ResNet, were used to compare the validity and accuracy of the model.
## Optimize
In order to optimize the model, the optimal parameters need to be found, namely the selection and tuning of the hyperparameters. Model selection was carried out based on accuracy and F1 scores, and several groups of models with the best results were selected, together with cross-validation and exhaustive enumeration to improve the confidence of the models. Separate predictions were made for X_test and the final classification was selected. Finally, our models achieved the desired accuracy and F1 score in the validation set.
