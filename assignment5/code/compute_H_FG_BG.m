function [H] = compute_H_FG_BG(TS_DCT, mu, sigma, pi_class, C, d)
    num_sample = size(TS_DCT, 1);
    H = zeros(num_sample, C);
    
    for i = 1:num_sample
        for j = 1:C
            H(i, j) = compute_gaussian(TS_DCT(i, :), mu(j, :), sigma(:,:,j), d) * pi_class(j, 1);
        end
        sum_den = sum(H(i,:), 'all');
        for j = 1:C
            H(i, j) = H(i, j) / sum_den;
        end
    end
end