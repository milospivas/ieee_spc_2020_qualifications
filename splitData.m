function [X_train, X_test] = splitData(X, pct)
%SPLITDATA Summary of this function goes here
%   Detailed explanation goes here
    M = size(X,1);
    M_train = floor(M * pct /100);
        
    rng(42); % For reproducibility
    X = X(randperm(M),:);
    
    X_train = X(1:M_train, :);
    X_test = X(M_train + 1 : end, :);
end

