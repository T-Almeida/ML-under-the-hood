% load house attributes and prices
houses=csvread("housing.csv");
X = [houses(:, 5) houses(:, 6) houses(:, 7)];%total_bedrooms,population,median_income
y = houses(:, 9);

%split training data from test data
[X_train,X_test,y_train,y_test] = split_test_train(X, y);

% slace features of the training samples
[X_train_scale,mean,sigma] = scaleFeatures(X_train);
 
% add intercept parameter to the training data
X_train_scale = [ones(size(X_train_scale,1),1) X_train_scale];

% train the model with gradient descend
thetaGD = gradientDescent(X_train_scale,y_train);

% predict agains test dataset
X_test_new= [ones(size(X_test,1),1) ((X_test.-mean)./sigma)]; 
result = computeCost(X_test_new,y_test,thetaGD);
result

%using normal equation
X = [ones(size(X_train,1),1) X_train];

thetaNE = normalEqn(X,y_train);
X_test_new= [ones(size(X_test,1),1) X_test]; 
result = computeCost(X_test_new,y_test,thetaNE);
result

