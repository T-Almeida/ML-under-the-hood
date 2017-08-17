function [a,z] = feedFoward(NN,X,Thetas)
  % Apply foward propagation throw the neural net :D

  a = cell(NN.num_layers,1);
  z = cell(NN.num_layers-1,1);
  m = size(X, 1);
  %input Layer
  a{1,1} = [ones(m, 1) X]; %Mx401
  z{1,1} = a{1,1} * Thetas{1,1}'; %z2 = z{1,1}
  %hidden layers
  for l = 2:(NN.num_layers-1)
    a{l,1} = [ones(m, 1) sigmoid(z{l-1,1})]; %Mx26
    z{l,1} = a{l,1} * Thetas{l,1}';
  end
  %output layer
  a{NN.num_layers,1} = sigmoid(z{NN.num_layers-1,1});
  
end