function  X_poly = polynomialRegression(X,degree=2)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

m = size(X,1); % number of training examples
X_poly = X;
for i = 1:degree-1
  X_poly = [X_poly (X_poly(:,i)).^(i+1)];
end

end
