function [theta, J_history] = gradientDescent(X, y, theta=rand(size(X,2),1), alpha=0.01, num_iters=1500)
%gradientDescent performs gradient descent to try learn optimal theta
%   
%   X - features MxN M = number of samples N = number of features
%   y - labels Mx1
%   theta - initial parameters values 
%   alpha - learning rate
%   num_iters - num iterations for gradient descent converge

  m = length(y); % number of training examples
  J_history = zeros(num_iters, 1);
  for iter = 1:num_iters
    % updates parameters for linear regression 
    % theta = theta - partial derivaty cost function computeCost
    theta = theta - (alpha*1/m*sum( (X*theta-y).*X ))';
  end

end
