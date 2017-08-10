%data
load("data_houses.txt")
X = data_houses(:, 1);
y = data_houses(:, 2);

%plot data
fprintf('Plot Data.\n');
figure;
plot(X, y, 'rx', 'MarkerSize', 10); % Plot the data
ylabel('Y - values');
xlabel('X - values'); 

fprintf('Program paused. Press enter to continue.\n');
pause;

%train the model with gradient descend
fprintf('Train the model with gradient descend.\n');
X = [ones(size(X,1),1) X];
%inilialize with bad theta
[theta, theta_J_history]= gradientDescent(X,y,0.002,2000,[-1;4]);
fprintf('Learn theta %f\n', theta);
fprintf('Program paused. Press enter to continue.\n');
pause;


fprintf('Visualizing J(theta_0, theta_1) ...\n')

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = computeCost(X, y, t);
    end
end

% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;


surf(theta0_vals, theta1_vals, J_vals)
hold on;
plot3(theta_J_history(1,:), theta_J_history(2,:), theta_J_history(3,:),'rx')

xlabel('\theta_0'); ylabel('\theta_1');



