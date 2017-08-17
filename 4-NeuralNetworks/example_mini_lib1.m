%Neural Network apply to IRIS dataset USING NN_MINI_LIB

% using some function from other folder
addpath("../1-Linear&PolynomialRegression");

% add my mini LIB
addpath("./NN_mini_lib");
rand("seed",42);
load iris_dataset.mat %load iris_dataset

X = iris_dataset(:,1:4);
y = iris_dataset(:,5);
y = 1.+y; %Adapte lables to NN
%y = 1 iris-setosa
%y = 2 iris-versicolor
%y = 3 iris-virginica

[X_train,X_test,y_train,y_test] = split_test_train(X,y,0.2);

m = size(X_train, 1);


fprintf('\nCreate Neural Network ...\n')

%% Setup Neural Network 
dim_layers = [size(X_train,2); %input
4; % hidden
3] % output
NN = createNeuralNetwork(dim_layers);
                          
fprintf('\nTraining Neural Network... \n')
lambda = 0;
NN = trainNeuralNetwork(NN,X_train,y_train,lambda,150);

fprintf('\nTest Neural Network ...\n')
pred = neuralNetworkPredict(NN,X_test);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);