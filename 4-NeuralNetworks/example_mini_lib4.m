%NeuralNetwork with and without dropout


% using some function from other folder
addpath("../1-Linear&PolynomialRegression");

% add my mini LIB
addpath("./NN_mini_lib");

load ex4data1.mat %load iris_dataset

rand("seed",42);

[X_train,X_test,y_train,y_test] = split_test_train(X,y,0.2);


m = size(X_train, 1);


fprintf('\nCreate Neural Network ...\n')

%% Setup Neural Network Very deep Neural NeuralNetwork
dim_layers = [size(X_train,2); %input 
50;
25;
10] % output
NN = createNeuralNetwork(dim_layers);
                          

%WITH DROP OUT

fprintf('\nTraining Neural Network with Dropout... \n')
lambda = 0;
dropout = 1;
learningRate = 2;
NN = trainNeuralNetwork(NN,X_train,y_train,lambda,dropout,learningRate,4000);

fprintf('\nTest on Training Neural Network ...\n')
pred = neuralNetworkPredict(NN,X_train);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_train)) * 100);
fprintf('\nTest on TestSet Neural Network ...\n')
pred = neuralNetworkPredict(NN,X_test);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);