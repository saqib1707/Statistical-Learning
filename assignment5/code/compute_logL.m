function [logL] = compute_logL(TS_DCT, mu, sigma, pi_class, h_class, C, d)
    num_sample = size(TS_DCT, 1);

    logL = 0;
    for i = 1:num_sample
        for j = 1:C
            PXGZ_xi_ej = compute_gaussian(TS_DCT(i,:), mu(j,:), sigma(:,:,j), d);
            logL = logL + h_class(i,j) * log(PXGZ_xi_ej * pi_class(j,1)); 
        end
    end
end