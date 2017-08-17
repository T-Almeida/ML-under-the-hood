function NN=createNeuralNetwork(dim_layers)
% dim_layers Layer x 1 vector each entry correspond to the number of units in a layer
  
  NN.layers = dim_layers;
  NN.num_layers = size(dim_layers,1);

  NN.parameters = [];
  for i=1:(NN.num_layers-1)
    NN.parameters = [NN.parameters; randInitializeWeights(dim_layers(i,1), dim_layers(i+1,1))(:)];
  end
  
end