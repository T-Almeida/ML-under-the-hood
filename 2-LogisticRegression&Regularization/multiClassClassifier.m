function [thetas] = multiClassClassifier(X, y, lambda=0, numIter = 3000,initial_theta=zeros(size(X, 2), 1))
  % Use one vs all stratagy
  % y - lables class should start with 0
  % X - prepare training data
  % lambda - regularization constant 0 = None
  
  n_classifier = max(y); 
  %  Set options for fminunc
  options = optimset('GradObj', 'on', 'MaxIter', numIter);
  thetas = zeros(size(X,2),n_classifier+1);
  for i = 0:n_classifier
    y_temp = y==i;
    thetas(:,i+1) = fminunc(@(t)(costFunctionReg(t, X, y_temp,lambda,@costFunction)), initial_theta, options);
  end
end
