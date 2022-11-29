function [prob_error_1, prob_error_2] = classify_FG_BG_MLE(img, seg_mask_gt, zigzag_pat_lin, TS_FG, TS_BG, strategy)
    [img_height, img_width] = size(img);

    % number of training samples in Di for FG and BG
    num_TS_FG = size(TS_FG, 1);
    num_TS_BG = size(TS_BG, 1);

    % compute the covariance matrix of X|Y,T equal to the sample covariance
    TS_mu_FG = mean(TS_FG);
    TS_FG_bar = TS_FG - TS_mu_FG;
    TS_sigma_FG = (transpose(TS_FG_bar) * TS_FG_bar) / num_TS_FG;
    TS_sigma_FG_inv = inv(TS_sigma_FG);
    
    TS_mu_BG = mean(TS_BG);
    TS_BG_bar = TS_BG - TS_mu_BG;
    TS_sigma_BG = (transpose(TS_BG_bar) * TS_BG_bar) / num_TS_BG;
    TS_sigma_BG_inv = inv(TS_sigma_BG);

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

    % for the MLE solution
    norm_const_FG = 1 / sqrt(power(2 * pi, d) * det(TS_sigma_FG));
    norm_const_BG = 1 / sqrt(power(2 * pi, d) * det(TS_sigma_BG));

    for i = 1:img_height
        for j = 1:img_width
    
            img_block_dct = dct2(img_pad(i:i+7,j:j+7));
            block_dct_vec(zigzag_pat_lin, 1) = img_block_dct(:);
    
            % MLE solution
            x_minus_mu_FG = block_dct_vec - transpose(TS_mu_FG);
            x_minus_mu_BG = block_dct_vec - transpose(TS_mu_BG);

            PXGY_x_FG = norm_const_FG * exp(-0.5 * transpose(x_minus_mu_FG) * TS_sigma_FG_inv * x_minus_mu_FG);
            PXGY_x_BG = norm_const_BG * exp(-0.5 * transpose(x_minus_mu_BG) * TS_sigma_BG_inv * x_minus_mu_BG);
            PX_x = PXGY_x_FG * PY_FG + PXGY_x_BG * PY_BG;

            PYGX_FG_x(i,j) = (PXGY_x_FG * PY_FG) / PX_x;
            PYGX_BG_x(i,j) = (PXGY_x_BG * PY_BG) / PX_x;
    
            if (PYGX_FG_x(i,j) > PYGX_BG_x(i,j))
                seg_mask_res(i,j) = 1;
            else
                seg_mask_res(i,j) = 0;
            end
        end
    end

    % compute prob of error for the MLE solution
    prob_error_1 = compute_prob_error(seg_mask_gt, seg_mask_res, PY_FG, PY_BG, 1);
    prob_error_2 = compute_prob_error(seg_mask_gt, seg_mask_res, PY_FG, PY_BG, 2);
    disp("Probability of Error (MLE Solution): "+prob_error_1);
end