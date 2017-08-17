%Neural Network apply to full MNISTI dataset 50000 images for training and 10000 for test

% using some function from other folder
addpath("../1-Linear&PolynomialRegression");

% add my mini LIB
addpath("./NN_mini_lib");

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
%% Setup Neural Network 
dim_layers = [size(X_train,2); %input
400; % hidden
25;
10] % output
NN = createNeuralNetwork(dim_layers);


fprintf('\nTraining Neural Network... \n')
lambda = 1;
dropout = 1;
NN = trainNeuralNetwork(NN,X_train,y_train,lambda,dropout,500);

fprintf('\nTest Neural Network ...\n')
pred = neuralNetworkPredict(NN,X_test);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);


fprintf('\nVisualizing Neural Network... \n')

Thetas = cell(NN.num_layers-1,1);

counter = 1;
for i=1:(NN.num_layers-1)
    Thetas{i,1} = reshape(NN.parameters(counter:(counter-1)+(NN.layers(i+1,1)) * (NN.layers(i,1) + 1)), ...
                 NN.layers(i+1,1), (NN.layers(i,1) + 1));
    counter = counter+((NN.layers(i+1,1)) * (NN.layers(i,1) + 1));
end

displayData(Thetas{1,1}(:, 2:end));

