function [mu, sigma, pi_class] = EM_algorithm(TS_DCT, C, d)
    num_sample = size(TS_DCT, 1);

    % randomly initialize the EM-algorithm parameters
    mu = rand(C, d);
    pi_class = rand(C, 1);
    sigma = zeros(d, d, C);
    for j = 1:C
        sigma(:, :, j) = diag(rand(d, 1));
    end

    % start the EM algorithm iteration
    epsilon = 1e-5;
    stop_thresh = 1e-3;
    
    max_itr = 20;
    itr = 0;
    prev_logL = 1;
    converge = 0;
    
    % estimating FG class parameters
    while converge == 0
        itr = itr + 1;
        disp(itr+"/"+max_itr);
    
        h_class = compute_H_FG_BG(TS_DCT, mu, sigma, pi_class, C, d);
        
        % sum for all training samples and all components dimensions
        sum_h_class = sum(h_class, 'all');
        assert(round(sum_h_class) == num_sample);
    
        % sum only for all the training samples but not along the components
        % dimension
        sum_h_class_j = transpose(sum(h_class, 1));   % C x 1
    
        % update pi values
        pi_class = sum_h_class_j / sum_h_class;   % C x 1
    
        % update mu values
        mu = (transpose(h_class) * TS_DCT) ./ sum_h_class_j;   % C x d
    
        % update sigma values
        for j = 1:C
            x_minus_mu = TS_DCT - mu(j, :);
            x_minus_mu_sq = x_minus_mu.^2;
            temp1 = (transpose(h_class(:, j)) * x_minus_mu_sq) / sum_h_class_j(j,1);
            temp1 = max(temp1, epsilon);
            sigma(:, :, j) = diag(temp1);
        end
    
        logL = compute_logL(TS_DCT, mu, sigma, pi_class, h_class, C, d);
    
        if (abs((logL - prev_logL)/prev_logL) < stop_thresh) || (itr > max_itr)
            converge = 1;
        end

        prev_logL = logL;
    end
end