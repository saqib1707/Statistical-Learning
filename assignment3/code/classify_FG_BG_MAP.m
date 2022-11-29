function [prob_error_1, prob_error_2] = classify_FG_BG_MAP(img, seg_mask_gt, zigzag_pat_lin, TS_FG, TS_BG, strategy, alpha)
    [img_height, img_width] = size(img);

    % number of training samples in Di for FG and BG
    num_TS_FG = size(TS_FG, 1);
    num_TS_BG = size(TS_BG, 1);

    if strategy == 1
        % load prior1.mat
        prior1 = load("../data/Prior_1.mat");
        mu_prior_mu_FG = prior1.mu0_FG;
        mu_prior_mu_BG = prior1.mu0_BG;
        mu_prior_sigma_weights = prior1.W0;

    elseif strategy == 2
        % load prior2.mat
        prior2 = load("../data/Prior_2.mat");
        mu_prior_mu_FG = prior2.mu0_FG;
        mu_prior_mu_BG = prior2.mu0_BG;
        mu_prior_sigma_weights = prior2.W0;
    end

    mu_prior_sigma = alpha * diag(mu_prior_sigma_weights);   % same for FG and BG classes
    mu_prior_sigma_inv = inv(mu_prior_sigma);

    % compute the covariance matrix of X|Y,T equal to the sample covariance
    TS_mu_FG = mean(TS_FG);
    TS_FG_bar = TS_FG - TS_mu_FG;
    TS_sigma_FG = (transpose(TS_FG_bar) * TS_FG_bar) / num_TS_FG;
    TS_sigma_FG_inv = inv(TS_sigma_FG);
    
    TS_mu_BG = mean(TS_BG);
    TS_BG_bar = TS_BG - TS_mu_BG;
    TS_sigma_BG = (transpose(TS_BG_bar) * TS_BG_bar) / num_TS_BG;
    TS_sigma_BG_inv = inv(TS_sigma_BG);
    
    % compute posterior mean and covariance for FG class
    mu_post_sigma_FG = inv(num_TS_FG * TS_sigma_FG_inv + mu_prior_sigma_inv);
    mu_post_mu_FG = transpose((num_TS_FG * TS_mu_FG * TS_sigma_FG_inv + mu_prior_mu_FG * mu_prior_sigma_inv) * mu_post_sigma_FG);
    
    % compute posterior mean and covariance for BG class
    mu_post_sigma_BG = inv(num_TS_BG * TS_sigma_BG_inv + mu_prior_sigma_inv);
    mu_post_mu_BG = transpose((num_TS_BG * TS_mu_BG * TS_sigma_BG_inv + mu_prior_mu_BG * mu_prior_sigma_inv) * mu_post_sigma_BG);
    
    % compute parameters of predictive distribution for FG and BG class
    mu_pred_FG = mu_post_mu_FG;
    mu_pred_BG = mu_post_mu_BG;
    
    sigma_pred_FG = TS_sigma_FG;
%     sigma_pred_FG = mu_post_sigma_FG + TS_sigma_FG;
    sigma_pred_FG_inv = inv(sigma_pred_FG);

    sigma_pred_BG = TS_sigma_BG;
%     sigma_pred_BG = mu_post_sigma_BG + TS_sigma_BG;
    sigma_pred_BG_inv = inv(sigma_pred_BG);
    
    % ML-estimate for class prior PY_FG and PY_BG
    num_TS = num_TS_FG + num_TS_BG;
    PY_FG = num_TS_FG / num_TS;   % P(Y = FG)
    PY_BG = num_TS_BG / num_TS;   % P(Y = BG)
    
    % classification of each pixel into FG and BG
    d = 64;                           % feature dimension
    block_dct_vec = zeros(d, 1);
    seg_mask_res = zeros(size(img));

    PYGX_FG_x = zeros(size(img));
    PYGX_BG_x = zeros(size(img));

    % pad test image with 7 layers to the right and bottom
    img_pad = img_padding(img);
    
    % predict cheetah image using BDR
    norm_const_FG = 1 / sqrt(power(2 * pi, d) * det(sigma_pred_FG));
    norm_const_BG = 1 / sqrt(power(2 * pi, d) * det(sigma_pred_BG));
    
    for i = 1:img_height
        for j = 1:img_width
    
            img_block_dct = dct2(img_pad(i:i+7,j:j+7));
            block_dct_vec(zigzag_pat_lin, 1) = img_block_dct(:);
    
            x_minus_mu_FG = block_dct_vec - mu_pred_FG;
            x_minus_mu_BG = block_dct_vec - mu_pred_BG;
    
            PXGYT_x_FG = norm_const_FG * exp(-0.5 * transpose(x_minus_mu_FG) * sigma_pred_FG_inv * x_minus_mu_FG);
            PXGYT_x_BG = norm_const_BG * exp(-0.5 * transpose(x_minus_mu_BG) * sigma_pred_BG_inv * x_minus_mu_BG);
            PX_x = PXGYT_x_FG * PY_FG + PXGYT_x_BG * PY_BG;

            PYGX_FG_x(i,j) = (PXGYT_x_FG * PY_FG) / PX_x;
            PYGX_BG_x(i,j) = (PXGYT_x_BG * PY_BG) / PX_x;
    
            if (PYGX_FG_x(i,j) > PYGX_BG_x(i,j))
                seg_mask_res(i,j) = 1;
            else
                seg_mask_res(i,j) = 0;
            end
        end
    end

    % compute prob of error for the Bayesian solution
    prob_error_1 = compute_prob_error(seg_mask_gt, seg_mask_res, PY_FG, PY_BG, 1);
    prob_error_2 = compute_prob_error(seg_mask_gt, seg_mask_res, PY_FG, PY_BG, 2);
    disp("Probability of Error (MAP Solution): "+prob_error_1);
    
%   whos;
    f = figure();
    ax = gca;
    ax.FontSize = 16;
    f.WindowState = 'maximized';
    subplot(2,2,1);
    imshow(seg_mask_gt);
    title('Ground-Truth Mask')
    
    subplot(2,2,2);
    imshow(seg_mask_res);
    title('Result: 64-feature Mask')
    % imwrite(seg_mask_res_64, "../plots/seg_mask_res_64.png");
    
    subplot(2,2,3);
    imshow(img);
    title('Original Image')
    % imwrite(seg_mask_res_8, "../plots/seg_mask_res_best_8.png");

    subplot(2,2,4);
    imshow(img_pad);
    title('Padded Image')
    % imwrite(seg_mask_res_8, "../plots/seg_mask_res_best_8.png");

    close all;
end