
# Project 2: Digit Recognition (Part I)
  In this project, we have been familiarized with the MNIST dataset for digit recognition, a popular task in computer vision. We have implemented a linear regression which turned out to be inadequate for this task. We have also learned how to use scikit-learn's SVM for binary  classification and multiclass classification. Then, we have implemented our own softmax regression using gradient descent. Finally, we have experimented with different hyperparameters, different labels and different features, including kernelized features.
  - part1/linear_regression.py where you will implement linear regression
  - part1/svm.py where you will implement support vector machine
  - part1/softmax.py where you will implement multinomial regression
  - part1/features.py where you will implement principal component analysis (PCA) dimensionality reduction
  - part1/kernel.py where you will implement polynomial and Gaussian RBF kernels
  - part1/main.py where you will use the code you write for this part of the project
  
  To get warmed up to the MNIST data set run python main.py. This file provides code that reads the data from mnist.pkl.gz by calling the function get_MNIST_data that is provided in utils.py. The call to get_MNIST_data returns Numpy arrays:
  - train_x : A matrix of the training data. Each row of train_x contains the features of one image, which are simply the raw pixel values flattened out into a vector of length  784=282 . The pixel values are float values between 0 and 1 (0 stands for black, 1 for white, and various shades of gray in-between).
  - train_y : The labels for each training datapoint, also known as the digit shown in the corresponding image (a number between 0-9).
  - test_x : A matrix of the test data, formatted like train_x.
  - test_y : The labels for the test data, which should only be used to evaluate the accuracy of different classifiers in your report.
  Next, we call the function plot_images to display the first 20 images of the training set.

# Project 3: Digit Recognition (Part II)
  - part2-nn/neural_nets.py in which we implement our first neural net from scratch
  - part2-mnist/nnet_fc.py where we will start using PyTorch to classify MNIST digits
  - part2-mnist/nnet_conv.py where we will use convolutional layers to boost performance
  - part2-twodigit/mlp.py and part2-twodigit/conv.py which are for a new, more difficult version of the MNIST dataset
