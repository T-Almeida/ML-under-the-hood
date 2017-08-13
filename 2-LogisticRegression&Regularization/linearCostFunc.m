function [J,grad] = linearCostFunc(theta,X, y)

m = length(y); % number of training examples


error=(X*theta-y);
J = 1/(2*m)*error'*error;

grad = (1/m.* sum((X*theta-y).*X))';

end
