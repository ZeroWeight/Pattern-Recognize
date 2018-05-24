function e = adaboost_error(X, y, k, a, d, alpha)
% adaboost_error: returns the final error rate of a whole adaboost
% 
% Input
%     X     : n * p matrix, each row a sample
%     y     : n * 1 vector, each row a label
%     k     : iter * 1 vector,  selected dimension of features
%     a     : iter * 1 vector, selected threshold for feature-k
%     d     : iter * 1 vector, 1 or -1
%     alpha : iter * 1 vector, weights of the classifiers
%
% Output
%     e     : error rate      

%%% Your Code Here %%%
M = length(k);
[n,p] = size(X); 
p = zeros(n,M);
for i=1:M
    if k(i) == 0
        break;
    end
    p(:,i) = ((X(:, k(i)) <= a(i)) - 0.5) * 2 * d(i); % predicted label
end
pret = p * alpha;
pre = ((pret > 0) - 0.5 )*2;
e = sum((pre ~= y))/n;
%%% Your Code Here %%%

end