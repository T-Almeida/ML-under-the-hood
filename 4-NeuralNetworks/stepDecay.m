function lr = stepDecay(epoch,initial_lrate,epochs_drop=10.0, drop_ratio=0.80)
   drop = drop_ratio;
   epochs_drop = epochs_drop;
   lr = initial_lrate * drop^(mod(epoch,epochs_drop)==0); 
end