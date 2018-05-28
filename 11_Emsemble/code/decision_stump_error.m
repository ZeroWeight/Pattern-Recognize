function e = decision_stump_error(X, y, k, a, d, w)
% decision_stump_error returns error of the given stump
%
% Input
%     X : n * p matrix, each row a sample
%     y : n * 1 vector, each row a label
%     k : selected dimension of features
%     a : selected threshold for feature-k
%     d : 1 or -1
%
% Output
%     e : number of errors of the given stump 

p = ((X(:, k) <= a) - 0.5) * 2 * d; % predicted label
e = sum((p ~= y) .* w);

end