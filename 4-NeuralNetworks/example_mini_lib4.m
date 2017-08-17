%NeuralNetwork with and without dropout


% using some function from other folder
addpath("../1-Linear&PolynomialRegression");

% add my mini LIB
addpath("./NN_mini_lib");

load iris_dataset.mat %load iris_dataset

X = iris_dataset(:,1:4);
y = iris_dataset(:,5);
y = 1.+y; %Adapte lables to NN
%y = 1 iris-setosa
%y = 2 iris-versicolor
%y = 3 iris-virginica

[X_train,y_train] = generateSamples(X,y,8000);

X_test=X;
y_test=y;

m = size(X_train, 1);


fprintf('\nCreate Neural Network ...\n')

%% Setup Neural Network Very deep Neural NeuralNetwork
dim_layers = [size(X_train,2); %input
10; % hidden
4;

3] % output
NN = createNeuralNetwork(dim_layers);
                          

%WITH DROP OUT

fprintf('\nTraining Neural Network with Dropout... \n')
lambda = 1;
dropout = 0.75;
NN = trainNeuralNetwork(NN,X_train,y_train,lambda,dropout,400);

fprintf('\nTest on Training Neural Network ...\n')
pred = neuralNetworkPredict(NN,X_train);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_train)) * 100);
fprintf('\nTest on TestSet Neural Network ...\n')
pred = neuralNetworkPredict(NN,X_test);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);