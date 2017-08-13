%linear regretion with regularization and using fminunc example
fprintf('Linear regretion with regularization and using fminunc example\n');

% Generating random data that follow quadratic function
m = 100
X = sort(6*rand(m,1)-4');
y = 2 .+ X .+ -0.5.*(X.^2) + randn(size(X,1),1);


%plot data
fprintf('Plot Data.\n');
figure;
plot(X, y, 'rx', 'MarkerSize', 10); % Plot the data
ylabel('Y - values');
xlabel('X - values'); 

fprintf('Program paused. Press enter to continue.\n');
pause;

X_poly = polynomialRegression(X);
X_poly = scaleFeatures(X_poly); %scaling to speed up GD
X_poly = [ones(size(X_poly,1),1) X_poly];

% Initialize fitting parameters
initial_theta = zeros(size(X_poly, 2), 1);

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 1500);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(linearCostFunc(t, X_poly, y)), initial_theta, options);
  
%plot data
fprintf('Plot Data.\n');
figure;
hold on;
plot(X, y, 'rx', 'MarkerSize', 10); % Plot the data
plot(X, X_poly*theta,'-');
ylabel('Y - values');
xlabel('X - values'); 

fprintf('Check the figure, result looks prety good\n');
fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('Try Simulate overfitting the data with high degree polynomial\n');
fprintf('Unfortunately the implementation of polynomialRegression is to naive to be able to use high-degree polynomial\n');

X_poly = polynomialRegression(X,5);
X_poly = scaleFeatures(X_poly); %scaling to speed up GD
X_poly = [ones(size(X_poly,1),1) X_poly];

% Initialize fitting parameters
initial_theta = zeros(size(X_poly, 2), 1);

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 1500);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(linearCostFunc(t, X_poly, y)), initial_theta, options);

%plot data
fprintf('Plot Data.\n');
figure;
hold on;
plot(X, y, 'rx', 'MarkerSize', 10); % Plot the data
plot(X, X_poly*theta,'-');
ylabel('Y - values');
xlabel('X - values'); 
theta
fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('Same situation but now with regularization\n');

lambda = 100;

% Initialize fitting parameters
initial_theta = zeros(size(X_poly, 2), 1);

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 1500);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X_poly, y,lambda,@linearCostFunc)), initial_theta, options);

%plot data
fprintf('Plot Data.\n');
figure;
hold on;
plot(X, y, 'rx', 'MarkerSize', 10); % Plot the data
plot(X, X_poly*theta,'-');
ylabel('Y - values');
xlabel('X - values'); 
theta

fprintf('Regularization result in more smooth polynomial\n');
fprintf('Program paused. Press enter to continue.\n');
pause;