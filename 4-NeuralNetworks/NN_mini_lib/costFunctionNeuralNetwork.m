function [J grad] = costFunctionNeuralNetwork(nn_params,NN, X, y,lambda,dropout)
% Train a NeuralNetwork represent in Struct NN with training data X an
Thetas = cell(NN.num_layers-1,1);

counter = 1;
for i=1:(NN.num_layers-1)
    Thetas{i,1} = reshape(nn_params(counter:(counter-1)+(NN.layers(i+1,1)) * (NN.layers(i,1) + 1)), ...
                 NN.layers(i+1,1), (NN.layers(i,1) + 1));
    counter = counter+((NN.layers(i+1,1)) * (NN.layers(i,1) + 1));
end

                 
% Setup some useful variables
m = size(X, 1);
         
yBinaryMatrix = sparse (1:rows (y), y, 1); %MxK

[a z] = feedFoward(NN,X,Thetas,dropout);


%regularization
lambdaJ = 0;
for l = 1:(NN.num_layers-1)
  lambdaJ = lambdaJ + sum((Thetas{l,1}(:,2:end).^2)(:));
end

lambdaJ = lambda/(2*m)*lambdaJ;

J = 1/m * sum((-yBinaryMatrix.*log(a{NN.num_layers,1}) - (1.-yBinaryMatrix).*log(1.-a{NN.num_layers,1}))(:)) + lambdaJ;


%compute grad
delta = cell(NN.num_layers-1,1);
delta{NN.num_layers-1,1} = a{NN.num_layers,1} - yBinaryMatrix;% MxK
debug = delta{NN.num_layers-1,1};
for i = fliplr(1:(NN.num_layers-2))
  delta{i,1} = (delta{i+1,1}*Thetas{i+1,1})(:,2:end) .* sigmoidGradient(z{i,1}); %Mx25
end


Theta_grad = cell(NN.num_layers-1,1);
grad = [];
for k = 1:(NN.num_layers-1)
  Theta_grad{k,1} = delta{k,1}'*a{k,1}./m;
  % apply gradient regularization
  Theta_grad{k,1}(:,2:end) = Theta_grad{k,1}(:,2:end) + lambda/m.*Thetas{k,1}(:,2:end);
  grad = [grad; Theta_grad{k,1}(:)];
end
%debug = Theta_grad{2,1};
end
