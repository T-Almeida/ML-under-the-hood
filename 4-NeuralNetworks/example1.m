%Neural Network apply to full MNISTI dataset 50000 images for training and 10000 for test

% using some function from other folder
addpath("../1-Linear&PolynomialRegression");

%% Initialization
clear ; close all; clc

%load data                          
X = loadMNISTImages('train-images.idx3-ubyte')';
y = loadMNISTLabels('train-labels.idx1-ubyte');
y(y==0)=10;%octave indices

rand ("seed", 42); % for NN hyperparameters comparation

X_train = X(1:50000,:);
X_test = X(50001:end,:);
y_train = y(1:50000,1);
y_test = y(50001:end,1);

m = size(X_train, 1);

%% Setup Neural Network 
input_layer_size  = size(X_train,2);  % Input Images of Digits
hidden_layer_size = 100;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)
                          
fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 150);

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_train, y_train, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
                 
                 
pred = predict(Theta1, Theta2, X_test);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));

