function [X_train, X_test, idx_train, idx_test] = splitData(X, pct)
%SPLITDATA Summary of this function goes here
%   Detailed explanation goes here
    M = size(X,1);
    M_train = floor(M * pct /100);
        
    rng(42); % For reproducibility
    idx = randperm(M);
    idx_train = idx(1:M_train);
    idx_test = idx(M_train + 1 : end);
    
    X_train = X(idx_train, :);
    X_test = X(idx_test, :);
end

