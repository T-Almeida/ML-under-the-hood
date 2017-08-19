function [theta,J_history]=gradientDescent(theta,computeGrad,learningRate=0.01,num_iter=150)
  J_history = [];
  min_learningRate = 0.01;
  %C = abs(-log(num_iter))+min_learningRate;
  alfa = learningRate;
  for i = 1:num_iter
    [J, grads] = computeGrad(theta);
    %alfa = (learningRate*(-log(i)+C));
    %alfa = learningRate*log(i);
    %alfa = e^(learningRate/i);
    %alfa = learningRate;
    alfa = stepDecay(i,alfa,100,0.98);
    theta = theta - alfa*grads;
    J_history = [J_history;J];
    fprintf('Iteration %d learningRate %d | Cost: %4.6e\r', i, alfa,J);
    fflush(stdout);
  end
end