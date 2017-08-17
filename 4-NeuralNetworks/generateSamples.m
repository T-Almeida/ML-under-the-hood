function [X_new,y_new] = generateSamples(X,y,n_samples=1000)
  n = size(X,2);
  X_new = zeros(n_samples,n);
  y_new = zeros(n_samples,1);
  max_label = max(y);
  
  samples_per_label = floor(n_samples/max_label);
  current_n_samples=1;
  
  for l=1:max_label
    temp_X = X(y==l,:);
    mean_X = mean(temp_X);
    sigma = std(temp_X);

    for i=1:n
      X_new(current_n_samples:(current_n_samples+samples_per_label-1),i)=...
            normrnd(mean_X(1,i),sigma(1,i),samples_per_label,1);
    end
    y_new(current_n_samples:(current_n_samples+samples_per_label-1),1) = l;
    current_n_samples = current_n_samples+samples_per_label;
  end

  X_new = X_new(1:(current_n_samples-1),:);
  y_new = y_new(1:(current_n_samples-1),1);
end