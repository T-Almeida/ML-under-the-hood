function [X_train,X_test,y_train,y_test] = split_test_train(X, y, test_ratio=0.2)

  m = length(y); % number of training examples
  index = randperm(m)';
  %refactor this
  test_size = floor(m*test_ratio);
  train_size = m-test_size;
  X_train = X(index(1:train_size,1));
  y_train = y(index(1:train_size,1));
  X_test = X(index(train_size+1:m,1));
  y_test = y(index(train_size+1:m,1));
  
end
