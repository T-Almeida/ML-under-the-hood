function NN=trainNeuralNetwork(NN,X,y,lambda=0,dropout=1,numIter=150)
% NN - NeuralNetwork structure 
% X data
% y labels
% lambda regularization factor
% number of Iterations
% return NeuralNetwork Structure trainned
  options = optimset('MaxIter', numIter);

  % Create "short hand" for the cost function to be minimized
  costFunction = @(p) costFunctionNeuralNetwork(p, ...
                                     NN, X, y, lambda,dropout);

  % Now, costFunction is a function that takes in only one argument (the
  % neural network parameters)
  [nn_params, cost] = fmincg(costFunction, NN.parameters, options);
  NN.parameters = nn_params;
  fprintf('\nTraining Complite with the final loss %f \n',cost(length(cost),1));
end