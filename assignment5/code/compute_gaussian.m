function [result] = compute_gaussian(x, mu, sigma, d)
    x = reshape(x, [d,1]);
    mu = reshape(mu, [d,1]);
    sigma = reshape(sigma, [d,d]);
    
    det_sigma = det(sigma);

%     if (det_sigma == 0)
%         diag_sigma = diag(sigma);
%         diag_sigma = max(diag_sigma, 1e-5);
%         sigma = diag(diag_sigma);
%         det_sigma = det(sigma);
%     end

    sigma_inv = inv(sigma);
    x_minus_mu = x - mu;

    result = (1 / sqrt(power(2*pi, d) * det_sigma)) * exp(-0.5 * transpose(x_minus_mu) * sigma_inv * x_minus_mu);
end