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

% load the training sample DCT matrix
TS_DCT = load("../data/TrainingSamplesDCT_8_new.mat");
TS_DCT_FG = TS_DCT.TrainsampleDCT_FG;
TS_DCT_BG = TS_DCT.TrainsampleDCT_BG;

% load the subset training samples DCT matrix
TS_DCT_subsets = load("../data/TrainingSamplesDCT_subsets_8.mat");

% get the D1 data
D1_FG = TS_DCT_subsets.D1_FG;
D1_BG = TS_DCT_subsets.D1_BG;

% get the D2 data
D2_FG = TS_DCT_subsets.D2_FG;
D2_BG = TS_DCT_subsets.D2_BG;

% get the D3 data
D3_FG = TS_DCT_subsets.D3_FG;
D3_BG = TS_DCT_subsets.D3_BG;

% get the D4 data
D4_FG = TS_DCT_subsets.D4_FG;
D4_BG = TS_DCT_subsets.D4_BG;

for strategy = 1:2
   for dataset = 1:4
% strategy = 1;
% dataset = 1;

if (dataset == 1)
    D_FG = D1_FG;
    D_BG = D1_BG;
elseif (dataset == 2)
    D_FG = D2_FG;
    D_BG = D2_BG;
elseif (dataset == 3)
    D_FG = D3_FG;
    D_BG = D3_BG;
elseif (dataset == 4)
    D_FG = D4_FG;
    D_BG = D4_BG;
end

% load alpha.mat and compute parameters for mean prior dist.
mu_prior_sigma_alpha = load("../data/Alpha.mat").alpha;
num_alpha_val = size(mu_prior_sigma_alpha, 2);

prob_error_1 = zeros(1, num_alpha_val);
prob_error_2 = zeros(1, num_alpha_val);

prob_error_1_MAP = zeros(1, num_alpha_val);
prob_error_2_MAP = zeros(1, num_alpha_val);

for k = 1:num_alpha_val
    alpha = mu_prior_sigma_alpha(1, k);

    fprintf("\n");
    disp("Strategy - "+strategy+"  ,  Dataset - D"+dataset+"  ,  Alpha: "+alpha);
    
    [prob_error_1(1,k), prob_error_2(1,k)] = classify_FG_BG_Bayesian(img, seg_mask_gt, zigzag_pat_lin, D_FG, D_BG, strategy, dataset, alpha);
    
    [prob_error_1_MAP(1,k), prob_error_2_MAP(1,k)] = classify_FG_BG_MAP(img, seg_mask_gt, zigzag_pat_lin, D_FG, D_BG, strategy, alpha);
end

[prob_error_1_MLE, prob_error_2_MLE] = classify_FG_BG_MLE(img, seg_mask_gt, zigzag_pat_lin, D_FG, D_BG, strategy);
% disp("Prob. of Error (MLE, Method-1): "+ prob_error_1_MLE);
% disp("Prob. of Error (MLE, Method-2): "+ prob_error_2_MLE);
prob_error_1_MLE_vec = ones(1, num_alpha_val)*prob_error_1_MLE;

lw = 2;
figure;
plot(mu_prior_sigma_alpha, prob_error_1, 'color', 'r', 'LineWidth', lw);
hold on;
plot(mu_prior_sigma_alpha, prob_error_1_MAP, 'color', 'b', 'LineWidth', lw);
hold on;
plot(mu_prior_sigma_alpha, prob_error_1_MLE_vec, 'color', 'g', 'LineWidth', lw);

set(gca, 'XScale', 'log');
ax.FontSize = 25;
xlabel("Alpha");
ylabel("Probability of Error");
title("Probability of Error vs Alpha: Strategy-"+strategy+", D"+dataset);
legend('Predictive Equation', 'MAP Solution', 'MLE Solution');
grid on;
% close all;
saveas(gcf, "../plots/S"+strategy+"_D"+dataset+"_prob_err_plot.png");

close;
   end
end