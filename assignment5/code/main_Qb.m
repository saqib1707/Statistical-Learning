clc; clear;
close all;

% load the original image and ground-truth segmentation mask
img = im2double(imread("../data/cheetah.bmp"));
img = img(:, 1:end-2);
seg_mask_gt = im2double(imread('../data/cheetah_mask.bmp'));
seg_mask_gt = seg_mask_gt(:, 1:end-2);

% load the zigzag pattern file
zigzag_pat = importdata("../data/zigzag_pattern.txt");
zigzag_pat_lin = zigzag_pat(:) + 1;   % adding 1 for converting to matlab indexes

[img_height, img_width] = size(img);

% pad test image with 7 layers to the right and bottom
img_pad = img_padding(img);

% compute the dct of the padded image once and reuse it for all test
dct_dim = 64;                           % feature dimension
dct_mat = zeros(img_height * img_width, dct_dim);
itr = 0;

for i = 1:img_height
    for j = 1:img_width
        itr = itr + 1;
        img_block_dct = dct2(img_pad(i:i+7, j:j+7));
        dct_mat(itr, zigzag_pat_lin) = img_block_dct(:);
    end
end

disp("Image DCT Computed !!!");

% load the training sample DCT matrix
TS_DCT = load("../data/TrainingSamplesDCT_8_new.mat");
TS_DCT_FG = TS_DCT.TrainsampleDCT_FG;
TS_DCT_BG = TS_DCT.TrainsampleDCT_BG;

% compute the prior class probabilities
num_sample_FG = size(TS_DCT_FG, 1);
num_sample_BG = size(TS_DCT_BG, 1);
num_sample = num_sample_FG + num_sample_BG;

PY_FG = num_sample_FG / num_sample;
PY_BG = num_sample_BG / num_sample;

d_EM = 64;     % feature size/dimension
C_list = [1,2,4,8,16,32];
C_list_len = size(C_list, 2);
dim_list = [1,2,4,8,16,24,32,40,48,56,64];
num_dim = size(dim_list, 2);

prob_error = zeros(C_list_len, num_dim);

for C_itr = 1:C_list_len

    C = C_list(1, C_itr);
    [mu_FG, sigma_FG, pi_FG] = EM_algorithm(TS_DCT_FG, C, d_EM);
    [mu_BG, sigma_BG, pi_BG] = EM_algorithm(TS_DCT_BG, C, d_EM);

    disp("EM - Parameters Estimation Complete - "+C_itr+"/"+C_list_len);

    for D_itr = 1:num_dim
        d = dim_list(1, D_itr);
        seg_mask_res = zeros(size(img));

        % compute gaussian formula for speed
        num_test_samples = size(dct_mat, 1);
        PXGY_x_FG = zeros(num_test_samples, 1);
        PXGY_x_BG = zeros(num_test_samples, 1);

        for k = 1:C
            term1_FG = 1 / sqrt(power(2 * pi, d) * det(sigma_FG(1:d,1:d,k)));
            x_minus_mu_FG = dct_mat(:,1:d) - mu_FG(k,1:d);
            sigma_FG_inv = inv(sigma_FG(1:d,1:d,k));
            PXGY_x_FG = PXGY_x_FG + term1_FG * exp(-0.5 * sum((x_minus_mu_FG * sigma_FG_inv) .* x_minus_mu_FG, 2)) * pi_FG(k,1);

            term1_BG = 1 / sqrt(power(2 * pi, d) * det(sigma_BG(1:d,1:d,k)));
            x_minus_mu_BG = dct_mat(:,1:d) - mu_BG(k,1:d);
            sigma_BG_inv = inv(sigma_BG(1:d,1:d,k));
            PXGY_x_BG = PXGY_x_BG + term1_BG * exp(-0.5 * sum((x_minus_mu_BG * sigma_BG_inv) .* x_minus_mu_BG, 2)) * pi_BG(k,1);
        end

        PX_x = PXGY_x_FG * PY_FG + PXGY_x_BG * PY_BG;

        PYGX_FG_x = (PXGY_x_FG * PY_FG) ./ PX_x;
        PYGX_BG_x = (PXGY_x_BG * PY_BG) ./ PX_x;

        itr = 0;
        for i = 1:img_height
            for j = 1:img_width
                itr = itr + 1;
                if (PYGX_FG_x(itr,1) > PYGX_BG_x(itr,1))
                    seg_mask_res(i,j) = 1;
                end
            end
        end

        % compute prob of error for the Mixture models
        prob_error(C_itr, D_itr) = compute_prob_error(seg_mask_gt, seg_mask_res, PY_FG, PY_BG, 1);
        disp("Prob. Error (d = "+d+"): "+prob_error(1, D_itr));
    end

    % plot the prob error VS dimension plot here
    figure;
    plot(dim_list, prob_error(C_itr,:), 'color', 'b', 'LineWidth', 2);

    ax.FontSize = 25;
    xlabel("dimension");
    ylabel("Prob. Error [P(error)]");
    title("Probability of Error vs Dimension: Classifier - (C = "+C+")");
    grid on;
    saveas(gcf, "../plots/Qb/C_C_"+C+"_prob_err_plot.png");
    close;
end

figure;
for C_itr = 1:C_list_len
    plot(dim_list, prob_error(C_itr,:), 'LineWidth', 2);
    hold on;
end

ax.FontSize = 25;
xlabel("dimension");
ylabel("Prob. Error [P(error)]");
title("Probability of Error vs Dimension: Classifier - (C = [1,2,4,8,16,32])");
legend("C=1","C=2","C=4","C=8","C=16","C=32");
grid on;
saveas(gcf, "../plots/Qb/C_C_all_prob_err_plot.png");
close;