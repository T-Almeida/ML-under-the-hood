function NN=trainNeuralNetwork(NN,X,y,lambda=0,dropout=1,learningRate=0.1,numIter=150)
% NN - NeuralNetwork structure 
% X data
% y labels
% lambda regularization factor
% number of Iterations
% return NeuralNetwork Structure trainned


  % Create "short hand" for the cost function to be minimized
  costFunction = @(p) costFunctionNeuralNetwork(p, ...
                                     NN, X, y, lambda,dropout);

     
  [nn_params, cost] = gradientDescent(NN.parameters,costFunction,learningRate,numIter);
  NN.parameters = nn_params;
  fprintf('\nTraining Complite with the final loss %f \n',cost(length(cost),1));
end