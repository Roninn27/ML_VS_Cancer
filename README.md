In this project, we used three files,eda.py, ml.py, and cnn.py

eda.py is some exploratory analysis of the data

ml.py is some tests of different classification subnetworks. If you want a single training model, then you need to comment out the other functions. The last model draws a comparison of all the models.

cnn has Alex and res18,If you want to look at a single model, you need to comment out one of them and train it.

All of our code is running on colab, please pay attention to the file path when using

INTRODUCTION
  The problem addressed in this project is the multiclassification problem, with models constructed in a data prediction manner. The main challenge is that this is not a traditional dichotomous problem, but a quadruple classification problem. Since the data are normalized, the mean and standard deviation are calculated to observe the approximate level of dispersion. In addition, the cell slice image datasets were npy files, which we chose to pre-process by converting it to jpg format.
  After the data was analysed, new challenges were identified. Due to the oversized images, random cropping was performed; again, due to the insufficient amount of data, data augmentation was used to perform feature spreading, data format transformation and data enhancement on the data. Several common classification learning algorithms were used to compare the effectiveness and accuracy of the models, including logistic regression, random forest, K-nearest neighbour, support vector machine, Gaussian Bayesian, decision tree classifier, AlexNet and ResNet.
In order to optimize the model, the optimal parameters need to be found, namely the selection and tuning of the hyperparameters. Model selection was carried out based on accuracy and F1 scores, and several groups of models with the best results were selected, together with cross-validation and exhaustive enumeration to improve the confidence of the models. Separate predictions were made for X_test and the final classification was selected. Finally, our models achieved the desired accuracy and F1 score in the validation set.
