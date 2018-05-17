function W = least_sq_multi(X, y, Lambda, w_0)
% least_sq_multi solves least square problems with different L1-norm
% penalty constants
%
% Inputs 
%     X      : n * M matrix, each row a sample with M features
%     y      : n * 1 vector, each element a target
%     Lambda : 1 * L vecotr, each element a L1-norm penalty constant
%     w_0    : M * 1 vector, the initial weight 
%
% Outputs
%     W      : M * L matrix, the column a weight vector

[~, M] = size(X);
L = length(Lambda);
W = zeros(M, L);

w_l = w_0;
for l = 1: L
  w_l = least_sq_L1(X, y, Lambda(l), w_l); 
  W(:,l) = w_l;
end

end