function [theta] = normalEqn(X, y, lambda=0)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.


lambdaMatrix = eye(size(X,2));
lambdaMatrix(1,1) = 0; % dont penalize theta(0) intercept term

theta=pinv(X'*X + lambda.*lambdaMatrix)*X'*y;

end
