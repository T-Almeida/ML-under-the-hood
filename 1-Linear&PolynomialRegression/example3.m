%
m = 100
X = 6*rand(m,1)-3';
y = 2 .+ X .+ 0.5.*(X.^2) + randn(size(X,1),1);

%plot data
fprintf('Plot Data.\n');
figure;
plot(X, y, 'rx', 'MarkerSize', 10); % Plot the data
ylabel('Y - values');
xlabel('X - values'); 

fprintf('Program paused. Press enter to continue.\n');
pause;

poly_X = polynomialRegression(X);
poly_X = [ones(size(poly_X,1),1) poly_X];
thetaNE = normalEqn(poly_X,y)
computeCost(poly_X,y,thetaNE)
fprintf('Predicted theta values are prety good.\n');
fprintf('Program paused. Press enter to continue.\n');
pause;

%using gradientDescend without scale
random_init_theta = randn(size(poly_X,2),1)

[thetaGD, theta_GD_history ]= gradientDescent(poly_X,y,0.01,1500,random_init_theta);
computeCost(poly_X,y,thetaGD)
fprintf('Program paused. Press enter to continue.\n');
pause;

%using gradientDescend with scale
poly_X = polynomialRegression(X);
[poly_X_scaled, mean,sigma]= scaleFeatures(poly_X);
poly_X_scaled = [ones(size(poly_X,1),1) poly_X_scaled];
[thetaGD2, theta_GD2_history ]= gradientDescent(poly_X_scaled,y,0.01,1500,random_init_theta);;
computeCost(poly_X_scaled,y,thetaGD2)

figure;
plot(1:size(theta_GD_history,2), theta_GD_history(4,:), '-b', 'LineWidth', 2);
hold on;
plot(1:size(theta_GD2_history,2), theta_GD2_history(4,:), '-g', 'LineWidth', 2);
legend ({"Not scaled", "Scaled"});
xlabel('Number of iterations');
ylabel('Cost J');

fprintf('Compare gradientDescent with scale features vs un scale features.\n');
fprintf('Program paused. Press enter to continue.\n');
pause;