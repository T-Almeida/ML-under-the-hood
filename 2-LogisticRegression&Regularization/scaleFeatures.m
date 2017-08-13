function [X_norm, mu, sigma] = scaleFeatures(X)
%scaleFeatures Normalizes the features in X 

  mu = mean(X);
  sigma = std(X);
  X_norm = (X.-mu)./sigma;

end
