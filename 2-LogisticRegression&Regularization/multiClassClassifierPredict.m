function [max_prob,acc] = multiClassClassifierPredict(thetas,X, y=-1)
  % Use one vs all stratagy
  % thetas - Nx(number classifiers)
  % X - prepare test data
  % y - compute acc not need
  %
  % Resturn
  % if y=-1 max_prob Mx2 1 - colum = probability of belonging to class of column 2
  % else    max_prob Mx3 where 3 colum is the right and wrong guesses
  % acc accuracy, only apply if y!=-1
  
  acc = -1; % invalid

  predictions = sigmoid(X*thetas);

  [v_max, indices] = max(predictions,[],2);
  class=indices-1;
  max_prob = [v_max class];
  if y ~=-1
    acc = sum(y==class)/length(y)*100;
    max_prob = [max_prob y==class];
  end
end
