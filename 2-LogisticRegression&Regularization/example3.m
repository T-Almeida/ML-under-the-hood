% using some function from other folder
addpath("../1-Linear&PolynomialRegression");

%Similar to example2, but now using functions to simplify and generalize multi class classification problem

load iris_dataset.mat %load iris_dataset

X = iris_dataset(:,1:4);
y = iris_dataset(:,5);
%y = 0 iris-setosa
%y = 1 iris-versicolor
%y = 2 iris-virginica

[X_train,X_test,y_train,y_test] = split_test_train(X,y,0.2);

X_train = [ones(size(X_train,1),1) X_train];
X_test = [ones(size(X_test,1),1) X_test];

% train
thetas = multiClassClassifier(X_train,y_train);

%predict
[result,acc] = multiClassClassifierPredict(thetas,X_test,y_test);