# -*- coding: utf-8 -*-
"""9417_proj

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yZH_HhkCcySNepikNmVWqU-SOtoSc9iV
"""

cd /content/drive/MyDrive/9417_proj


from __future__ import print_function, division
from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import matplotlib.pylab as plt
import itertools
import time

import sklearn
from sklearn import model_selection
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

import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from __future__ import print_function, division
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
from sklearn.metrics import confusion_matrix, f1_score,accuracy_score

import os
import shutil
import time
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.models as models
import matplotlib.pyplot as plt
from itertools import chain

# The conversion to Image is run only once to generate the image, and the folder needs to be created before the image is generated
X_train = np.load("X_train.npy", mmap_mode='r')
y_train = np.load("y_train.npy")
X_train, X_test, Y_train, Y_test = train_test_split(X_train, y_train, test_size=0.2)

matplotlib.use('agg')
plt.figure(figsize=(12, 12))
for i in range(len(X_train)):
    plt.imshow(X_train[i])
    plt.savefig(f'/content/drive/MyDrive/9417_proj/Dataset/train/{int(Y_train[i])}/image{str(i+1)}.jpeg', dpi=300)
    plt.savefig(f'/content/drive/MyDrive/9417_proj/Dataset/train/{int(Y_train[i])}/image{str(i+1)}.png', dpi=300)
    plt.savefig(f'/content/drive/MyDrive/9417_proj/Dataset/train/{int(Y_train[i])}/image{str(i+1)}.jpg', dpi=300)
    plt.clf()
    plt.close()
for i in range(len(X_test)):
    plt.imshow(X_test[i])
    # plt.savefig(f'/content/drive/MyDrive/9417_proj/Dataset/val/{int(Y_test[i])}/image{str(i+1)}.jpeg', dpi=300)
    plt.savefig(f'/content/drive/MyDrive/9417_proj/Dataset/val/{int(Y_test[i])}/image{str(i+1)}.png', dpi=300)
    # plt.savefig(f'/content/drive/MyDrive/9417_proj/Dataset/val/{int(Y_test[i])}/image{str(i+1)}.jpg', dpi=300)
    plt.clf()
    plt.close()

# License: BSD
# Author: Sasank Chilamkurthy

cudnn.benchmark = True
plt.ion()   # interactive mode
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/content/drive/MyDrive/9417_proj/Dataset'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=10,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(dataset_sizes)
print(len(dataloaders['val']))
print(len(dataloaders['train']))

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    result = dict()
    result['train'] = defaultdict(list)
    result['val'] = defaultdict(list)


    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
   
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            result[phase]['loss'].append(epoch_loss)
            result[phase]['Acc'].append(float(epoch_acc))
            

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts,'resnet_model.pth')
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,result

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                plt.imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
print(num_ftrs)


def loss_acc_curve(result,name,e):
  plt.figure(figsize=(15, 8), dpi=120)
  plt.subplot(1,2,1)
  plt.plot(range(0,e),result['train']['Acc'],c='blue',label ='train')
  plt.plot(range(0,e),result['val']['Acc'],c='orange',label ='test')
  plt.title('acc_curve')
  plt.xlabel('Number of epochs')
  plt.ylabel('Acc')
  plt.legend(loc ='best')
  plt.subplot(1,2,2)
  plt.plot(range(0,e),result['train']['loss'],c='g',label ='tarin')
  plt.plot(range(0,e),result['val']['loss'],c='r',label ='test')
  plt.title('loss_curve')
  plt.xlabel('Number of epochs')
  plt.ylabel('loss')
  plt.legend(loc = 'best')
  plt.savefig(f'{name}.png')
  plt.show

"""AlexNet"""

model_alex = models.alexnet(pretrained=True)

model_alex.classifier[0] = nn.Dropout(p=0.3)
model_alex.classifier[3] = nn.Dropout(p=0.3)
model_alex.classifier[6] = nn.Linear(4096, 4)

print(model_alex.classifier)

model_alex = model_alex.to(device)
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_alex.parameters(), lr=0.0001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)


model_alex,result = train_model(model_alex, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=20)
loss_acc_curve(result,'alex_20_0.001',20)

"""AlexNet"""



"""Res18"""

# Here the size of each output sample is set to 4.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 4)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9,weight_decay=0.0005,nesterov=True)
# optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0002, amsgrad=False)
# Decay LR by a factor of 0.1 every 10 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)


model_ft,result = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=30)
loss_acc_curve(result,'res18_50_SGD_10',50)

"""Res18"""


import torch
import numpy as np
import torchvision.models as models
from sklearn.metrics import confusion_matrix, f1_score,accuracy_score

import os
import shutil
import time
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.models as models
import matplotlib.pyplot as plt
from itertools import chain
 

# Model weights and category labels
weight_path = 'resnet_model.pth'
 
classes = ['0', '1', '2', '3']
 
# Draws the obfuscation matrix function
def plot_confusion_matrix(classes, cm, savename, title='Confusion Matrix'):
    plt.figure(figsize=(15, 12), dpi=200)
    np.set_printoptions(precision=2)
 
    # The probability value of each cell in the confusion matrix
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.0f" % (c,), color='red', fontsize=15, va='center', ha='center')
 
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
 
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
 
    # Display confusion matrix
    #plt.show()
    plt.savefig(savename, format='jpg')


def LoadNet(weight_path):
    net = models.resnet18(pretrained=False)
    fc_features = net.fc.in_features
    net.fc = torch.nn.Linear(fc_features, len(classes))
    net.load_state_dict(torch.load(weight_path))
    net.eval()
    net.cuda()
    return net
 
# Import the test data set
#testset, test_loader = data_test(config.save_test_dir)
#print('\nThe test data is loaded\n')
 
true_label = []
pred_label = []
# x_test = np.load("X_test.npy")
# Load model
model = LoadNet(weight_path)

for i, (input, label) in enumerate(dataloaders['val']):
    input, label = input.to(device), label.to(device)
    output = model(input)
    pred = output.data.max(1, keepdim=True)[1]
    prediction = pred.squeeze(1)
    prediction = prediction.cpu().numpy()
    label = label.cpu().numpy()
    print('The real label of the input image is:{}, The prediction label is:{}'.format(label, prediction))
    true_label.append(list(label))
    pred_label.append(list(prediction))
# Calculate the confusion matrix and plot
true_label=list(chain.from_iterable(true_label))
pred_label = list(chain.from_iterable(pred_label))
#plot_confusin_pictures(true_label, pred_label)
print(pred_label)
print(true_label)

# print('f1-score is: ', f1_score(true_label, pred_label, average = 'macro'))
print('f1-score is: ', f1_score(true_label, pred_label, average = 'weighted'))
print('Accuracy score is: ', accuracy_score(true_label, pred_label))
cm = confusion_matrix(true_label, pred_label)
plot_confusion_matrix(classes, cm, 'confusion_matrix_res.jpg', title='confusion matrix')

x_test = np.load("X_test.npy")
matplotlib.use('agg')
plt.figure(figsize=(12, 12))
for i in range(len(x_test)):
    plt.imshow(x_test[i])
    plt.savefig(f'/content/drive/MyDrive/9417_proj/testset/test/00000{i+1}', dpi=300)
    plt.clf()
    plt.close()

image_index = [1, 10, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
               11, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
               12, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
               13, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
               14, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
               15, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
               16, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169,
               17, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
               18, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189,
               19, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199,
               2, 20, 200, 201, 202, 203, 204, 205, 206,207, 208, 209,
               21, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
               22, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229,
               23, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
               24, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249,
               25, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259,
               26, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269,
               27, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279,
               28, 280, 281, 282, 283, 284, 285, 286, 287,
               29,
               3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
               4, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
               5, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
               6, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
               7, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
               8, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
               9, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
print(len(image_index))

data_dir = '/content/drive/MyDrive/9417_proj/Valset'
image_datasets = datasets.ImageFolder(os.path.join(data_dir), data_transforms['val'])
dataloaders_test =torch.utils.data.DataLoader(image_datasets, batch_size=4, num_workers=4)
dataset_sizes = len(image_datasets)
class_names = image_datasets.classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pred_label = []
print(dataloaders_test)
for i, (input,label) in enumerate(dataloaders_test): 
    #input = torch.stack(input, dim=1)
    #label = label.to(device)
    input= input.to(device)
    output = model(input)
    pred = output.data.max(1, keepdim=True)[1]
    prediction = pred.squeeze(1)
    prediction = prediction.cpu().numpy()
    pred_label.append(list(prediction))

pred_label = list(chain.from_iterable(pred_label))    
print(image_index)
ture = [-1] * 287
# print(len(ture))
# print(len(pred_label))
for i in range(287):
  # print(image_index[i])
  # print(pred_label[i])
  ture[image_index[i]-1] = pred_label[i]

pred_np = np.array(ture)
print(pred_np)
np.save("final_pred.npy",pred_np)

from collections import Counter
Counter(ture)

