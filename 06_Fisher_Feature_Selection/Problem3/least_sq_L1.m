function w = least_sq_L1(X, y, lambda, w_0)
% least_sq_L1 solves the least square problem with L1-norm penalty
%
% Inputs 
%     X      : n * M matrix, each row a sample with M features
%     y      : n * 1 vector, each element a target
%     lambda : the penalty constant
%     w_0    : M * 1 vector, the initial weight 
%
% Outputs
%     w      : M * 1 vector, the weight vector

[n, M] = size(X); % n samples, each with M features

% precompute a_k's, since they don't vary when w is updated
a = sum(X .* X)' / n; % M * 1 vector

w = w_0;
iter = 0;
err_tol = 1e-8;
w_new = w;
while (1)
  max_err = 0;
  for k = 1:M
    % evaluate c_k based on w and X 
    psi = X(:,k);
    res = y - X * w + w(k) * psi;
    c_k = mean(psi .* res);
    
    % update w(k)
    w_new(k) = sign(c_k) * max(abs(c_k)-lambda,0) / a(k);
    max_err = norm(w_new - w,Inf);
    w = w_new;
  end
  iter = iter + 1;
  if (max_err < err_tol) 
      return; 
  end

end