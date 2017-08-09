% load house attributes and prices
load("data_houses.txt")
X = data_houses(:, 1:2);
y = data_houses(:, 3);

% slace features of the training samples
[X,mean,sigma] = scaleFeatures(X);
 
% add intercept parameter to the training data
X = [ones(size(X,1),1) X];

% train the model with gradient descend
thetaGD = gradientDescent(X,y);

% predict house with 1600m2 and 3 bedrooms
X_new_sample = [1600 1];
X_new= [1 scaleFeatures(X_new_sample)]; 
result = X_new * thetaGD;
result

%using normal equation
X = [ones(size(data_houses,1),1) data_houses(:, 1:2)];

thetaNE = normalEqn(X,y);
X_new= [1 X_new_sample]; 
result = X_new * thetaNE;
result

