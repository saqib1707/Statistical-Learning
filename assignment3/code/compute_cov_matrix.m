function [cov_mat] = compute_cov_matrix(X, X_mu)
    N = size(X, 1);
    
    cov_mat = zeros(64);
    for i = 1:N
        term = transpose(X(i,:) - X_mu);
%         disp(size(term));
        cov_mat = cov_mat + term * transpose(term);
    end
    cov_mat = cov_mat / N;
end