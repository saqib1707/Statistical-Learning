clc; clear;
close all;

% load the original image and ground-truth segmentation mask
img = im2double(imread("../data/cheetah.bmp"));
seg_mask_gt = im2double(imread('../data/cheetah_mask.bmp'));
% imwrite(seg_mask_gt, '../plots/seg_mask_gt.png');
% imwrite(img, '../plots/original_img.png');
[img_height,img_width] = size(img);

% load the zigzag pattern file
zigzag_pat = importdata("../data/zigzag_pattern.txt");
zigzag_pat_lin = zigzag_pat(:)+1;   % adding 1 for converting to matlab indexes

% load the training sample DCT matrix
train_sample_DCT = load("../data/TrainingSamplesDCT_8_new.mat");
train_sample_DCT_FG = train_sample_DCT.TrainsampleDCT_FG;
train_sample_DCT_BG = train_sample_DCT.TrainsampleDCT_BG;

% compute the index histograms for FG and BG training samples
% FG_feature = find_feature(train_sample_DCT_FG);   % P(x|cheetah)
% BG_feature = find_feature(train_sample_DCT_BG);   % P(x|grass)

% prior probabilities for cheetah and grass
num_sample_FG = size(train_sample_DCT_FG,1);
num_sample_BG = size(train_sample_DCT_BG,1);
total_samples = num_sample_FG + num_sample_BG;

% compute the prior probabilities using MLE
PY_FG = num_sample_FG/total_samples;   % P(Y=cheetah)
PY_BG = num_sample_BG/total_samples;   % P(Y=grass)
disp("Maximum Likelihood estimate for the Prior Probabilities:");
disp("P_Y(i = FG) = "+PY_FG);
disp("P_Y(i = BG) = "+PY_BG);
fprintf('\n');

% plotting the prior probabilities for both classes in a histogram format
figure;
X = categorical({'FG (cheetah)', 'BG (grass)'});
Y = [PY_FG PY_BG];
h_prior = bar(X,Y,0.5);
% legend('PX|Y(x|cheetah)', );
ax = gca;
ax.FontSize = 16;
xlabel('class');
ylabel('P_{Y}(i)');
title('Histogram of Prior Probabilities');
grid on;
% saveas(gcf, "../plots/hist_prior_prob.png");
close;


% Question-2

% compute MLE for the parameters of the CCD PX|Y(x|FG) and P_X|Y(x|BG)
% The parameters are mu_FG, mu_BG, sigma_FG, sigma_BG
feature_dim = size(train_sample_DCT_FG,2);

% compute MLE estimates for the mean for FG and BG classes
mu_FG = transpose(mean(train_sample_DCT_FG,1));
mu_BG = transpose(mean(train_sample_DCT_BG,1));

% compute MLE estimates for the covariance matrix for FG and BG classes
sample_mean_sub_FG = (train_sample_DCT_FG - transpose(mu_FG));  % num_sample_FG x 64
sample_mean_sub_BG = (train_sample_DCT_BG - transpose(mu_BG));  % num_sample_BG x 64

sigma_FG = zeros(feature_dim);
for i=1:num_sample_FG
    sigma_FG = sigma_FG + reshape(sample_mean_sub_FG(i,:), [feature_dim,1])*reshape(sample_mean_sub_FG(i,:), [1,feature_dim]);
end
sigma_FG = sigma_FG/num_sample_FG;

sigma_BG = zeros(feature_dim);
for i=1:num_sample_BG
    sigma_BG = sigma_BG + reshape(sample_mean_sub_BG(i,:), [feature_dim,1])*reshape(sample_mean_sub_BG(i,:), [1,feature_dim]);
end
sigma_BG = sigma_BG/num_sample_BG;

disp("Maximum Likelihood Estimate for the parameters of CCD using Gaussian Assumption");
disp("MLE Mean vector for FG class: ");
disp(mu_FG);
disp("MLE Mean vector for BG class: ");
disp(mu_BG);
disp("MLE Covariance Matrix for FG class:");
disp(sigma_FG);
disp("MLE Covariance Matrix for BG class:");
disp(sigma_BG);


% plotting 64 marginals
for i=1:feature_dim
    mu_FG_i = mu_FG(i,1);
    mu_BG_i = mu_BG(i,1);
    sigma_FG_i = sigma_FG(i,i);
    sigma_BG_i = sigma_BG(i,i);

    mean_FG_BG = (mu_FG_i + mu_BG_i)/2;
    max_std_FG_BG = max(sigma_FG_i, sigma_BG_i);

    x_range = [mean_FG_BG - 10*max_std_FG_BG:(20*max_std_FG_BG)/1000:mean_FG_BG + 10*max_std_FG_BG];

    y_val_FG = zeros(size(x_range));
    y_val_BG = zeros(size(x_range));

    % computing univariate gaussian density for both classes
    temp1_FG = (1/(sqrt(2*pi*sigma_FG_i^2)));
    temp1_BG = (1/(sqrt(2*pi*sigma_BG_i^2)));
    temp2_FG = -1/(2*sigma_FG_i^2);
    temp2_BG = -1/(2*sigma_BG_i^2);
    
    j = 1;
    for x=x_range
        y_val_FG(1,j) = temp1_FG*exp(temp2_FG*((x-mu_FG_i)^2));
        y_val_BG(1,j) = temp1_BG*exp(temp2_BG*((x-mu_BG_i)^2));
        j = j+1;  
    end
    
    figure;
    plot(x_range, y_val_FG); 
    hold on;
    plot(x_range, y_val_BG);
    ax = gca;
    ax.FontSize = 16;
    xlabel("X\_"+{i}+" (DCT coeff)");
    ylabel("P_{X\_"+{i}+"|Y} (x | i)");
    title('Marginal Densities');
    legend('FG (cheetah)','BG (grass)');
    grid on;

    saveas(gcf, "../plots/marginals/FG_BG_marginal_"+i+".png");
    close;
end

% Question - 3 - compute the Bayesian Decision Rule and Classification
% Now, take the image => pad the image => compute the DCT coeffs for each adjacent 8x8
% block => form the 64-dim feature vector => classify that feature vector
% into {FG,BG}

% classification of each pixel into FG and BG
img_dct = zeros(size(img));
block_dct_vec = zeros(feature_dim,1);
seg_mask_res_64 = zeros(size(img));

% pad test image with 7 layers to the right and bottom
img_pad = img(:,:);
img_pad(end+1:end+7,:) = img(end-7:end-1,:);
img_pad(1:end-7,end+1:end+7) = img(:,end-7:end-1);
img_pad(end-7:end,end-7:end) = img(end-7:end,end-7:end);

% compute sigma_FG and sigma_BG determinant and inverse
sigma_FG_det = det(sigma_FG);
sigma_BG_det = det(sigma_BG);
sigma_FG_inv = inv(sigma_FG);
sigma_BG_inv = inv(sigma_BG);

PYGX_FG_x_64 = zeros(size(img));
PYGX_BG_x_64 = zeros(size(img));

temp1_FG = 1/(sqrt(power(2*pi,feature_dim)*sigma_FG_det));
temp1_BG = 1/(sqrt(power(2*pi,feature_dim)*sigma_BG_det));

count_FG_pixels = 0;
count_BG_pixels = 0;
count_FG_error = 0;
count_BG_error = 0;

for i=1:img_height
    for j=1:img_width
        img_block_dct = dct2(img_pad(i:i+7,j:j+7));
        block_dct_vec(zigzag_pat_lin,1) = img_block_dct(:);

        dct_mean_sub_FG = (block_dct_vec - mu_FG);
        dct_mean_sub_BG = (block_dct_vec - mu_BG);

        PXGY_x_FG = temp1_FG*exp(-0.5*transpose(dct_mean_sub_FG)*sigma_FG_inv*dct_mean_sub_FG);
        PXGY_x_BG = temp1_BG*exp(-0.5*transpose(dct_mean_sub_BG)*sigma_BG_inv*dct_mean_sub_BG);
        PX_x = PXGY_x_FG*PY_FG + PXGY_x_BG*PY_BG;

        PYGX_FG_x_64(i,j) = (PXGY_x_FG*PY_FG)/PX_x;
        PYGX_BG_x_64(i,j) = (PXGY_x_BG*PY_BG)/PX_x;

        if (PYGX_FG_x_64(i,j) > PYGX_BG_x_64(i,j))
            seg_mask_res_64(i,j) = 1;
        else
            seg_mask_res_64(i,j) = 0;
        end

        % computing probability of error
        if (seg_mask_gt(i,j) == 1)
            count_FG_pixels = count_FG_pixels + 1;
            if (seg_mask_res_64(i,j) == 0)
                count_FG_error = count_FG_error + 1;
            end
        elseif (seg_mask_gt(i,j) == 0)
            count_BG_pixels = count_BG_pixels + 1;
            if (seg_mask_res_64(i,j) == 1)
                count_BG_error = count_BG_error + 1;
            end
        end
    end
end
assert(count_FG_pixels == sum(seg_mask_gt, 'all'));
prob_error_64 = (count_FG_error/count_FG_pixels)*PY_FG + (count_BG_error/count_BG_pixels)*PY_BG;

% Now, take the image => pad the image => compute the DCT coeffs for each adjacent 8x8
% block => form the 64-dim feature vector => re-orient using zigzag pattern => extract
% the best 8 feature => classify that feature vector into {FG,BG}
feature_dim = 8;
seg_mask_res_8 = zeros(size(img));
PYGX_FG_x_8 = zeros(size(img));
PYGX_BG_x_8 = zeros(size(img));

best_8_feat_index = [22,29,30,38,39,44,45,47];
worst_8_feat_index = [2,3,5,6,7,9,12,15];

% compute sigma_FG and sigma_BG determinant and inverse
sigma_FG_8 = sigma_FG(best_8_feat_index, best_8_feat_index);
sigma_BG_8 = sigma_BG(best_8_feat_index, best_8_feat_index);

sigma_FG_det_8 = det(sigma_FG_8);
sigma_BG_det_8 = det(sigma_BG_8);
sigma_FG_inv_8 = inv(sigma_FG_8);
sigma_BG_inv_8 = inv(sigma_BG_8);

temp1_FG = 1/(sqrt(power(2*pi,feature_dim)*sigma_FG_det_8));
temp1_BG = 1/(sqrt(power(2*pi,feature_dim)*sigma_BG_det_8));

count_FG_pixels = 0;
count_BG_pixels = 0;
count_FG_error = 0;
count_BG_error = 0;

for i=1:img_height
    for j=1:img_width
        img_block_dct = dct2(img_pad(i:i+7,j:j+7));
        block_dct_vec(zigzag_pat_lin,1) = img_block_dct(:);

        block_dct_vec = block_dct_vec(best_8_feat_index,1);
        dct_mean_sub_FG = (block_dct_vec - mu_FG(best_8_feat_index,1));
        dct_mean_sub_BG = (block_dct_vec - mu_BG(best_8_feat_index,1));

        PXGY_x_FG = temp1_FG*exp(-0.5*transpose(dct_mean_sub_FG)*sigma_FG_inv_8*dct_mean_sub_FG);
        PXGY_x_BG = temp1_BG*exp(-0.5*transpose(dct_mean_sub_BG)*sigma_BG_inv_8*dct_mean_sub_BG);
        PX_x = PXGY_x_FG*PY_FG + PXGY_x_BG*PY_BG;
            
        PYGX_FG_x_8(i,j) = (PXGY_x_FG*PY_FG)/PX_x;
        PYGX_BG_x_8(i,j) = (PXGY_x_BG*PY_BG)/PX_x;

        if (PYGX_FG_x_8(i,j) > PYGX_BG_x_8(i,j))
            seg_mask_res_8(i,j) = 1;
        else
            seg_mask_res_8(i,j) = 0;
        end

        % computing probability of error
        if (seg_mask_gt(i,j) == 1)
            count_FG_pixels = count_FG_pixels + 1;
            if (seg_mask_res_8(i,j) == 0)
                count_FG_error = count_FG_error + 1;
            end
        elseif (seg_mask_gt(i,j) == 0)
            count_BG_pixels = count_BG_pixels + 1;
            if (seg_mask_res_8(i,j) == 1)
                count_BG_error = count_BG_error + 1;
            end
        end
    end
end

assert(count_FG_pixels == sum(seg_mask_gt, 'all'));
prob_error_8 = (count_FG_error/count_FG_pixels)*PY_FG + (count_BG_error/count_BG_pixels)*PY_BG;

f = figure();
ax = gca;
ax.FontSize = 16;
f.WindowState = 'maximized';
subplot(2,2,1);
imshow(seg_mask_gt);
title('Ground-Truth Mask')

subplot(2,2,2);
imshow(seg_mask_res_64);
title('Result: 64-feature Mask')
imwrite(seg_mask_res_64, "../plots/seg_mask_res_64.png");

subplot(2,2,3);
imshow(seg_mask_res_8);
title('Result: 8-feature Mask')
imwrite(seg_mask_res_8, "../plots/seg_mask_res_best_8.png");


% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
%         if (seg_mask_res_64(i,j) == 0 && seg_mask_gt(i,j) == 1)
% %             prob_error_64 = prob_error_64 + PYGX_BG_x_64(i,j)*PX_x;
%             prob_error_64 = prob_error_64 + 0;
%         elseif (seg_mask_res_64(i,j) == 0 && seg_mask_gt(i,j) == 1)
%             prob_error_64 = prob_error_64 + PYGX_BG_x_64(i,j)*PX_x;
%         elseif (seg_mask_res_64(i,j) == 0 && seg_mask_gt(i,j) == 0)
% %             prob_error_64 = prob_error_64 + PYGX_FG_x_64(i,j)*PX_x;
%               prob_error_64 = prob_error_64 + 0;
%         elseif (seg_mask_res_64(i,j) == 1 && seg_mask_gt(i,j) == 0)
%             prob_error_64 = prob_error_64 + PYGX_FG_x_64(i,j)*PX_x;
%         end


% FG
% for i=1:64
%     figure;
%     h_marginal_FG = histogram(train_sample_DCT_FG(:,i), Normalization="probability");
%     h_marginal_BG = histogram(train_sample_DCT_BG(:,i), Normalization="probability");
%     % legend('PX|Y(x|cheetah)');
%     ax = gca;
%     ax.FontSize = 12;
%     xlabel("X\_"+{i}+" (DCT coeff)");
%     ylabel("P_{X\_"+{i}+"|Y} (x|FG)");
%     title('Marginal Densities');
%     grid on;
%     saveas(gcf, "../plots/FG_marginal_"+i+".png");
%     close;
% end
% close all;

% BG
% for i=1:64
%     figure;
%     h_marginal = histogram(train_sample_DCT_BG(:,i), Normalization="probability");
%     % legend('PX|Y(x|cheetah)');
%     ax = gca;
%     ax.FontSize = 12;
%     xlabel("X\_"+{i}+" (DCT coeff)");
%     ylabel("P_{X\_"+{i}+"|Y} (x|BG)");
%     title('Marginal Densities');
%     grid on;
%     saveas(gcf, "../plots/BG_marginal_"+i+".png");
%     close;
% end
% close all;
