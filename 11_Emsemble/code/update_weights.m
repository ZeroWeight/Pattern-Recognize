function w_update = update_weights(X, y, k, a, d, w, alpha)
% update_weights update the weights with the recent classifier
% 
% Input
%     X        : n * p matrix, each row a sample
%     y        : n * 1 vector, each row a label
%     k        : selected dimension of features
%     a        : selected threshold for feature-k
%     d        : 1 or -1
%     w        : n * 1 vector, old weights
%     alpha    : weights of the classifiers
%
% Output
%     w_update : n * 1 vector, the updated weights

%%% Your Code Here %%%
p = ((X(:, k) <= a) - 0.5) * 2 * d; % predicted label
w_update = w .* exp((p ~= y) .* alpha);
w_update = w_update ./ (sum(w_update));
%%% Your code Here %%%

end