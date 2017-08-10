function [theta, theta_J_history] = gradientDescent(X, y, alpha=0.01, num_iters=1500,theta=randn(size(X,2),1))
%gradientDescent performs batch gradient descent to learn optimal theta of a convex cost function
%   
%   X - features MxN M = number of samples N = number of features
%   y - labels Mx1
%   theta - initial parameters values 
%   alpha - learning rate
%   num_iters - num iterations for gradient descent converge

  m = length(y); % number of training examples
  theta_J_history = zeros(size(X,2)+1, num_iters);
  for iter = 1:num_iters
    % updates parameters for linear regression 
    % theta = theta - partial derivaty cost function computeCost
    theta = theta - (alpha*1/m*sum( (X*theta-y).*X ))';
    theta_J_history(:,iter)=[theta;computeCost(X,y,theta)];
  end
  
end
