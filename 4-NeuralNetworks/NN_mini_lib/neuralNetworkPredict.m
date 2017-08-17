function p=neuralNetworkPredict(NN,X)
  %NN - neural network structure
  %X - data to apply the inference
 
  Thetas = cell(NN.num_layers-1,1);

  counter = 1;
  for i=1:(NN.num_layers-1)
      Thetas{i,1} = reshape(NN.parameters(counter:(counter-1)+(NN.layers(i+1,1)) * (NN.layers(i,1) + 1)), ...
                   NN.layers(i+1,1), (NN.layers(i,1) + 1));
      counter = counter+((NN.layers(i+1,1)) * (NN.layers(i,1) + 1));
  end

  a = feedFoward(NN,X,Thetas);

  [dummy, p] = max(a{NN.num_layers,1} , [], 2);

end