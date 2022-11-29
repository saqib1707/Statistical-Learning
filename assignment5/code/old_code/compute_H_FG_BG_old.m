function [h_FG, h_BG] = compute_H_FG_BG(TS_DCT_FG, mu_FG, sigma_FG, pi_FG, TS_DCT_BG, mu_BG, sigma_BG, pi_BG, C, d)
    num_sample_FG = size(TS_DCT_FG, 1);
    num_sample_BG = size(TS_DCT_BG, 1);
    
    h_FG = zeros(num_sample_FG, C);
    h_BG = zeros(num_sample_BG, C);
    
    for i = 1:num_sample_FG
        for j = 1:C
            h_FG(i, j) = compute_gaussian(TS_DCT_FG(i, :), mu_FG(j, :), sigma_FG(:,:,j), d) * pi_FG(j, 1);
        end
        sum_den = sum(h_FG(i,:), 'all');
        for j = 1:C
            h_FG(i, j) = h_FG(i, j) / sum_den;
        end
    end
    
    for i = 1:num_sample_BG
        for j = 1:C
            h_BG(i, j) = compute_gaussian(TS_DCT_BG(i, :), mu_BG(j, :), sigma_BG(:,:,j), d) * pi_BG(j, 1);
        end
        sum_den = sum(h_BG(i,:), 'all');
        for j = 1:C
            h_BG(i, j) = h_BG(i, j) / sum_den;
        end
    end
end