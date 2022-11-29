clc; clear;
close all;

% load the original image and ground-truth segmentation mask
img = im2double(imread("../data/cheetah.bmp"));
img = img(:, 1:end-2);
seg_mask_gt = im2double(imread('../data/cheetah_mask.bmp'));
seg_mask_gt = seg_mask_gt(:, 1:end-2);

[img_height, img_width] = size(img);

% load the zigzag pattern file
zigzag_pat = importdata("../data/zigzag_pattern.txt");
zigzag_pat_lin = zigzag_pat(:) + 1;   % adding 1 for converting to matlab indexes

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

C = 8;   % number of components

% randomly initialize the EM-algorithm parameters
d = 64;     % feature size/dimension
mu_FG = rand(C, d);
% mu_FG_temp = zeros(C, d);
pi_FG = rand(C, 1);
sigma_FG = zeros(d, d, C);
% sigma_FG_temp = zeros(d, d, C);
for j = 1:C
    sigma_FG(:, :, j) = diag(rand(d, 1));
end

mu_BG = rand(C, d);
pi_BG = rand(C, 1);
sigma_BG = zeros(d, d, C);
for j = 1:C
    sigma_BG(:, :, j) = diag(rand(d, 1));
end

% start the EM algorithm iteration
epsilon = 1e-2;
max_itr = 100;
itr = 0;
prev_logL = 0;

% estimating FG class parameters
while converge == 0
    itr = itr + 1;
    disp(itr+"/"+max_itr);

    [h_FG, h_BG] = compute_H_FG_BG(TS_DCT_FG, mu_FG, sigma_FG, pi_FG, TS_DCT_BG, mu_BG, sigma_BG, pi_BG, C, d);
    
    % sum for all training samples and all components dimensions
    sum_h_FG = sum(h_FG, 'all');
    assert(round(sum_h_FG) == num_sample_FG);

    % sum only for all the training samples but not along the components
    % dimension
    sum_h_FG_j = transpose(sum(h_FG, 1));   % C x 1

    % update pi values
    pi_FG = sum_h_FG_j / sum_h_FG;   % C x 1

    % update mu values
    mu_FG = (transpose(h_FG) * TS_DCT_FG) ./ sum_h_FG_j;   % C x d

    % update sigma values
    for j = 1:C
        x_minus_mu = TS_DCT_FG - mu_FG(j, :);
        x_minus_mu_sq = x_minus_mu.^2;
        temp1 = max((transpose(h_FG(:, j)) * x_minus_mu_sq) / sum_h_FG_j(j,1), epsilon);
        sigma_FG(:, :, j) = diag(temp1);
    end

    logL = compute_logL(TS_DCT_FG, mu_FG, sigma_FG, pi_FG, C, d);

    if abs((logL - prev_logL)/prev_logL) < 0.1 || itr > max_itr
        converge = 1;
    end
    prev_logL = logL;
end

% estimating BG class parameters
while converge == 0
% for itr = 1:max_itr
    itr = itr + 1;
    disp(itr+"/"+max_itr);
    % compute h_ij
    [h_FG, h_BG] = compute_H_FG_BG(TS_DCT_FG, mu_FG, sigma_FG, pi_FG, TS_DCT_BG, mu_BG, sigma_BG, pi_BG, C, d);
    
    % sum for all training samples and all components dimensions
    sum_h_FG = sum(h_FG, 'all');
    sum_h_BG = sum(h_BG, 'all');
    assert(round(sum_h_FG) == num_sample_FG);
    assert(round(sum_h_BG) == num_sample_BG);

    % sum only for all the training samples but not along the components
    % dimension
    sum_h_FG_j = transpose(sum(h_FG, 1));   % C x 1
    sum_h_BG_j = transpose(sum(h_BG, 1));   % C x 1

    % update pi values
    pi_FG = sum_h_FG_j / sum_h_FG;   % C x 1
    pi_BG = sum_h_BG_j / sum_h_BG;   % C x 1

    % update mu values
%     mu_FG = transpose(transpose(TS_DCT_FG) * h_FG) ./ sum_h_FG_j;   % C x d
%     mu_BG = transpose(transpose(TS_DCT_BG) * h_BG) ./ sum_h_BG_j;   % C x d

    mu_FG = (transpose(h_FG) * TS_DCT_FG) ./ sum_h_FG_j;   % C x d
    mu_BG = (transpose(h_BG) * TS_DCT_BG) ./ sum_h_BG_j;   % C x d

    % alternative method for calculating mu_FG
%     for j = 1:C
%         sum_temp = zeros(1,d);
%         sum_h = 0;
%         for i = 1:num_sample_FG
%             sum_temp = sum_temp + h_FG(i, j) * TS_DCT_FG(i,:);
%             sum_h = sum_h + h_FG(i,j);
%         end
%         mu_FG_temp(j,:) = sum_temp / sum_h;
%     end
%     disp("Difference between short and long version of mu calc: "+sum(abs(mu_FG - mu_FG_temp), 'all'));

    % update sigma values
    for j = 1:C
        x_minus_mu = TS_DCT_FG - mu_FG(j, :);
        x_minus_mu_sq = x_minus_mu.^2;
        temp1 = max((transpose(h_FG(:, j)) * x_minus_mu_sq) / sum_h_FG_j(j,1), epsilon);
        sigma_FG(:, :, j) = diag(temp1);

        x_minus_mu = TS_DCT_BG - mu_BG(j, :);
        x_minus_mu_sq = x_minus_mu.^2;
        temp1 = max((transpose(h_BG(:, j)) * x_minus_mu_sq) / sum_h_BG_j(j,1), epsilon);
        sigma_BG(:, :, j) = diag(temp1);
    end


%     for j = 1:C
%         x_minus_mu = TS_DCT_FG - mu_FG(j, :);
%         sigma_FG(:,:,j) = ((transpose(x_minus_mu) .* transpose(h_FG(:, j))) * x_minus_mu) / sum_h_FG_j(j, 1);
% 
%         x_minus_mu = TS_DCT_BG - mu_BG(j, :);
%         sigma_BG(:,:,j) = ((transpose(x_minus_mu) .* transpose(h_BG(:, j))) * x_minus_mu) / sum_h_BG_j(j, 1);
%     end

    % alternative way to compute the sigma values
%     for j = 1:C
%         sum_h = 0;
%         for i = 1:num_sample_FG
%             x_minus_mu = reshape((TS_DCT_FG(i,:) - mu_FG(j,:)), [d,1]);
%             sigma_FG_temp(:,:,j) = reshape(sigma_FG_temp(:,:,j), [d,d]) + h_FG(i,j) * x_minus_mu * transpose(x_minus_mu);
%             sum_h = sum_h + h_FG(i,j);
%         end
%         sigma_FG_temp(:,:,j) = sigma_FG_temp(:,:,j) / sum_h;
%     end
%     disp("Difference between short and long version of sigma calc: "+sum(abs(sigma_FG - sigma_FG_temp), 'all'));

    logL = compute_logL(TS_DCT_FG, mu_FG, sigma_FG, pi_FG, TS_DCT_BG, mu_BG, sigma_BG, pi_BG, C, d);

    if abs((logL - prev_logL)/prev_logL) < 0.1 || itr > max_itr
        converge = 1;
    end
    prev_logL = logL;
end

disp('Done with EM');

dim_list = [1,2,4,8,16,24,32,40,48,56,64];

% classification of each pixel into FG and BG
d = 64;                           % feature dimension
block_dct_vec = zeros(d, 1);
seg_mask_res = zeros(size(img));

PYGX_FG_x = zeros(size(img));
PYGX_BG_x = zeros(size(img));

% pad test image with 7 layers to the right and bottom
img_pad = img_padding(img);

for i = 1:img_height
    for j = 1:img_width

        img_block_dct = dct2(img_pad(i:i+7,j:j+7));
        block_dct_vec(zigzag_pat_lin, 1) = img_block_dct(:);

        PXGY_x_FG = 0;
        PXGY_x_BG = 0;

        for k = 1:C
            PXGY_x_FG = PXGY_x_FG + compute_gaussian(block_dct_vec, mu_FG(k,:), sigma_FG(:,:,k), d) * pi_FG(k, 1);
            PXGY_x_BG = PXGY_x_BG + compute_gaussian(block_dct_vec, mu_BG(k,:), sigma_BG(:,:,k), d) * pi_BG(k, 1);
        end
        
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

% compute prob of error for the Mixture models
prob_error = compute_prob_error(seg_mask_gt, seg_mask_res, PY_FG, PY_BG, 1);
% prob_error_2 = compute_prob_error(seg_mask_gt, seg_mask_res, PY_FG, PY_BG, 2);
disp("Probability of Error: "+prob_error);