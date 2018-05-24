function [k, a, d] = decision_stump(X, y, w)
% decision_stump returns a rule ...
% h(x) = d if x(k) ? a, -d otherwise, %% change the encoding of '?'
%
% Input
%     X : n * p matrix, each row a sample
%     y : n * 1 vector, each row a label
%     w : n * 1 vector, each row a weight
%
% Output
%     k : the optimal dimension
%     a : the optimal threshold
%     d : the optimal d, 1 or -1

% total time complexity required to be O(p*n*logn) or less
%%% Your Code Here %%%
[n,p]=size(X);
sample = zeros(n,1);
e =inf;
k = 0;
dmat = zeros(1,p);
amat =zeros(1,p);
%total: O(pnlogn)
for m=1:p   % looping every feature of X, O(p) complexity
    ef=inf; 
    sample = unique(sort(X(:,m))); % sort the feature, O(nlogn) complexity
    for i=1:(n-1) % O(n) complexity
        temp = decision_stump_error(X, y, m, (sample(i)+sample(i+1))/2, 1, w);
        if ef > temp
            ef = temp;
            dmat(m) = 1;
            amat(m) = (sample(i)+sample(i+1))/2;
        end
        temp = decision_stump_error(X, y, m, (sample(i)+sample(i+1))/2, -1, w);
        if ef > temp
            ef = temp;
            dmat(m) = -1;
            amat(m) = (sample(i)+sample(i+1))/2;
        end
    end
    if e >ef
        e = ef;
        k = m;
end
a = amat(k);
d = dmat(k);
%%% Your Code Here %%%
end