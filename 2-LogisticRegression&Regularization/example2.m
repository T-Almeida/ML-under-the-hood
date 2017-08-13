% using some function from other folder
addpath("../1-Linear&PolynomialRegression");

% one vs all classifier
load iris_dataset.mat %load iris_dataset

X = iris_dataset(:,1:4);
y = iris_dataset(:,5);
%y = 0 iris-setosa
%y = 1 iris-versicolor
%y = 2 iris-virginica

[X_train,X_test,y_train,y_test] = split_test_train(X,y,0.2);

X_train = [ones(size(X_train,1),1) X_train];
X_test = [ones(size(X_test,1),1) X_test];

%prepare 3 classifiers
y0 = y_train==0;
y1 = y_train==1;
y2 = y_train==2;

% Initialize fitting parameters
initial_theta = zeros(size(X_train, 2), 1);

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 3000);
lambda = 0;
%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta_y0, cost_y0] = fminunc(@(t)(costFunctionReg(t, X_train, y0,lambda,@costFunction)), initial_theta, options);
[theta_y1, cost_y1] = fminunc(@(t)(costFunctionReg(t, X_train, y1,lambda,@costFunction)), initial_theta, options);
[theta_y2, cost_y2] = fminunc(@(t)(costFunctionReg(t, X_train, y2,lambda,@costFunction)), initial_theta, options);


% predict and test
predictions = sigmoid([X_test*theta_y0 X_test*theta_y1 X_test*theta_y2]);

[v_max, indices] = max(predictions,[],2);
indices=indices-1;
acc = sum(y_test==indices)/length(y_test)*100;
fprintf('The classifier have acc of: %.0f\n', acc);

%plotData

