% using some function from other folder
addpath("../1-Linear&PolynomialRegression");
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

%% Setup Neural Network 
dim_layers = [size(X_train,2); %input
4; % hidden
3] % output
NN = createNeuralNetwork(dim_layers);

%%OLD WAY

% Unroll parameters
initial_nn_params = NN.parameters;

%  You should also try different values of lambda
lambda = 1;

[debug J grad] = costFunctionNeuralNetwork(initial_nn_params, ...
                                   NN, X_train, y_train, lambda);

J
grad;