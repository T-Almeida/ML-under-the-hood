# ML-under-the-hood

This repository have implementation of various machine learning algorithms, in folder 4 is an implementaion of the library to manipulate fully connected neural networks

Many code and concepts follow [Stanford Machine Learning Class](https://www.coursera.org/learn/machine-learning), so if you are taking the class follow stanfor honor code. But as you can see some examples are different and represent different problems

## Stanford Honor Code

> "We strongly encourage students to form study groups,  and discuss the lecture videos (including in-video questions). We also encourage you to get together with friends to watch the videos together as a group. However,  the answers that you submit for the review questions should be your own work. For the programming exercises,  you are welcome to discuss them with other students,  discuss specific algorithms,  properties of algorithms,  etc.; we ask only that you not look at any source code written by a different student,  nor show your solution code to other students."

Files starter with "ex(.*)" corresponde to stanford ml class exercisces and may have been MODIFIED by me to fit the new examples I added.

Files starter with "example(.*)" corresponde new examples added by me. 

What the folder contains:

1 Linear regression and polymonial regression
 * Mean square error (costFunction)
 * Features scaling (mean and sigma method)
 * Polynomial Features generation (polynomial regression)
 * Gradient descent 
 * NormalEquation with regularization
 * Split traning set's in test and training set's
 * Ex and Examples

2 Logistic regression and Regularization (uses code from 1)
 * Sigmoid function
 * Logistic regression cost function with regularization
 * Linear Cost Function with regularization
 * Multi class classifier train (one vs all strategy)
 * Multi class classifier prediction (one vs all strategy)
 * Ex and Examples
 
3 Multi-Class and Neural Networks
 * More Multi class classifier train and prediction
 * Tested with image (digits) dataset (prety cool)
 * Neural Network froward propagation (prediction)
 * Ex and Examples

4 Neural Networks
 * Training neural network with backpropagation
 * Generate synthetic samples from semantic knowledge (in this case mean and std from Data :(, better case would be from ontology )
 * Manual method of compute partial derivatives to check if backpropagation is well implemented
 * Full MNIST data set 60000 samples (50000 for trainig 10000 for test)
 * "Mini" Library for create fully connected neural networks of any size (NN_mini_lib)
 	- Create fully connected Neural Network 
 	- Train that Neural Network
	- Train with dropout possibility (but some erros with fmin function waiting fix)
 	- Make predictions (classification problems) with the trained Neural network
 * Ex and Examples, "example_mini_lib" correspond to example using the library
 * "example_mini_lib3" train and test one Neural Network that only have 1.78% error on MNIST data set, since we are not using advanced techniques like batch normalization, dropout, more sophisticated activation functions this isn't a bad result at all :P
