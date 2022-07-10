import numpy as np
import matplotlib.pylab as plt
import itertools
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA, PCA
from sklearn.datasets import make_circles
import numpy as np


def image(X_train, y_train):
    plt.figure(figsize=(30, 20))
    num_0 = 0
    num_1 = 1 * 6
    num_2 = 2 * 6
    num_3 = 3 * 6
    for i in range(800):
        if y_train[i] == 0 and num_0 < 6:
            num_0 += 1
            print(num_0)
            plt.subplot(4, 6, num_0)
            plt.title("Class = %d" % y_train[i])
            plt.imshow(X_train[i])
        if y_train[i] == 1 and num_1 < 2 * 6:
            num_1 += 1
            print(num_1)
            plt.subplot(4, 6, num_1)
            plt.title("Class = %d" % y_train[i])
            plt.imshow(X_train[i])
        if y_train[i] == 2 and num_2 < 3 * 6:
            num_2 += 1
            print(num_2)
            plt.subplot(4, 6, num_2)
            plt.title("Class = %d" % y_train[i])
            plt.imshow(X_train[i])
        if y_train[i] == 3 and num_3 < 4 * 6:
            print('可以用', i)
            num_3 += 1
            print(num_3)

            plt.subplot(4, 6, num_3)
            plt.title("Class = %d" % y_train[i])
            plt.imshow(X_train[i])

        if num_0 == 6 and num_1 == 2 * 6 and num_2 == 3 * 6 and num_3 == 4 * 6:
            break

    plt.savefig('Classification of the image.png', dpi=400)
    plt.show()


def statistic_analysys(X_train, y_train, X_test):
    print('X_images.shape =', X_train.shape)
    print('X_images.min/mean/std/max = %.2f/%.2f' % (X_train.mean(), X_train.std()))
    print()
    print('Y_images.shape =', y_train.shape)
    print('Y_images.min/mean/std/max = %.2f/%.2f' % (y_train.mean(), y_train.std()))
    print()
    print('Y_images.shape =', X_test.shape)
    print('Y_images.mean/std = %.2f/%.2f' % (X_test.mean(), X_test.std()))


def describeData(a, b):
    # Total number of data images
    total = len(a)
    print('Total number of images: {}'.format(total))

    n0 = np.sum(b == 0)
    n1 = np.sum(b == 1)
    n2 = np.sum(b == 2)
    n3 = np.sum(b == 3)
    print('Number of Class0 Images: {}'.format(np.sum(b == 0)))
    print('Number of Class1 Images: {}'.format(np.sum(b == 1)))
    print('Number of Class2 Images: {}'.format(np.sum(b == 2)))
    print('Number of Class3 Images: {}'.format(np.sum(b == 3)))

    plt.bar(['class0', 'class1', 'class2', 'class3'], [n0, n1, n2, n3], color='lightblue')
    plt.title('The Numcer Of Different Class')
    plt.savefig('fenbutu', dpi=300)
    plt.show()

    # The proportion of images whose IDC is 1 is displayed
    print('Percentage of Class0 images: {:.2f}%'.format(n0 / total))
    print('Percentage of Class1 images: {:.2f}%'.format(n1 / total))
    print('Percentage of Class2 images: {:.2f}%'.format(n2 / total))
    print('Percentage of Class3 images: {:.2f}%'.format(n3 / total))
    plt.pie(x=[n0 / total, n1 / total, n2 / total, n3 / total], labels=['class0', 'class1', 'class2', 'class3'],
            autopct='%.2f%%')
    plt.legend(loc=(1, 0.01))
    plt.title('The Ratio Of Different Class')
    plt.savefig('fenbutu pie', dpi=300)
    plt.show()
    # Length, width, and number of channels of the data sample
    print('Image shape (Width, Height, Channels): {}'.format(a[0].shape))


def plotHistogram(a, b):
    plt.figure(figsize=(10, 5))

    # The first diagram draws the selected image
    plt.subplot(1, 2, 1)
    plt.imshow(a)
    plt.axis('off')
    plt.title(f'Class{int(y_train[b])}')

    # The second drawing draws a three-channel bar chart corresponding to the selected image
    histo = plt.subplot(1, 2, 2)
    histo.set_ylabel('Count')  # Set the y
    histo.set_xlabel('Pixel Intensity')  # Set the x axis
    n_bins = 30

    # Draw a bar chart

    # Bar graph plotting R channel
    plt.hist(a[:, :, 0].flatten(), bins=n_bins, lw=0, color='r', alpha=0.5);
    # Histogram plots G channel
    plt.hist(a[:, :, 1].flatten(), bins=n_bins, lw=0, color='g', alpha=0.5);
    # The bar chart plots channel B
    plt.hist(a[:, :, 2].flatten(), bins=n_bins, lw=0, color='b', alpha=0.5);
    plt.savefig(f'class{int(y_train[b])} three-channel.png', dpi=600)
    plt.show()


def pca(X_train, y_train):
    pca = PCA(n_components=2)
    projected = pca.fit_transform(X_train[0][0])
    plt.scatter(projected[:, 0], projected[:, 1], c=X_train[0][0][0:, 0])
    plt.show()


if __name__ == "__main__":
    X_train = np.load("X_train.npy", mmap_mode='r')
    y_train = np.load("y_train.npy")
    X_test = np.load("X_test.npy", mmap_mode='r')

    # image(X_train,y_train)
    # statistic_analysys(X_train,y_train,X_test)
    describeData(X_train, y_train)
    # plotHistogram(X_train[0],0)
    pca(X_train, y_train)
