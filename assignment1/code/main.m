clc; clear;
img = im2double(imread("../data/cheetah.bmp"));
[img_height,img_width] = size(img);
% imwrite(img, '../plots/original_img.png');

% load the zigzag pattern file
zigzag_pat = importdata("../data/zigzag_pattern.txt");
zigzag_pat_lin = zigzag_pat(:)+1;   % adding 1 for converting to matlab indexes

% figure;
% imshow(img);

% load the training sample DCT matrix
train_sample_DCT = load("../data/TrainingSamplesDCT_8.mat");
train_sample_DCT_FG = train_sample_DCT.TrainsampleDCT_FG;
train_sample_DCT_BG = train_sample_DCT.TrainsampleDCT_BG;

% compute the index histograms for FG and BG training samples
FG_feature = find_feature(train_sample_DCT_FG);   % P(x|cheetah)
BG_feature = find_feature(train_sample_DCT_BG);   % P(x|grass)

figure;
h_FG = histogram(FG_feature, 64, 'BinWidth', 1, 'Normalization', 'probability', 'BinEdges', 0:1:64);
% legend('PX|Y(x|cheetah)');
ax = gca;
ax.FontSize = 12;
xlabel('Feature X (1 <= x <= 64)');
ylabel('P_{X|Y}(x|cheetah)');
title('Histogram of CCD');
% saveas(gcf, "../plots/FG_hist.png");
PXGY_x_C = h_FG.Values;

figure;
h_BG = histogram(BG_feature, 64, 'BinWidth', 1, 'Normalization', 'probability', 'BinEdges', 0:1:64);
% legend('PX|Y(x|cheetah)');
ax = gca;
ax.FontSize = 12;
xlabel('Feature X (1 <= x <= 64)');
ylabel('P_{X|Y}(x|grass)');
title('Histogram of CCD');
% saveas(gcf, "../plots/BG_hist.png");
PXGY_x_G = h_BG.Values;

% prior probabilities for cheetah and grass
num_sample_FG = size(train_sample_DCT_FG,1);
num_sample_BG = size(train_sample_DCT_BG,1);
total_samples = num_sample_FG + num_sample_BG;

PY_C = num_sample_FG/total_samples;   % P(Y=cheetah)
PY_G = num_sample_BG/total_samples;   % P(Y=grass)

PXY_x_C = PXGY_x_C*PY_C;   % joint probability of X and Y
PXY_x_G = PXGY_x_G*PY_G;

PX_x = PXY_x_C + PXY_x_G;   % marginalization in Y to obtain P(X)

% classification of each pixel into cheetah and grass
img_dct = zeros(size(img));
block_dct_vec = zeros(1,64);
seg_mask_res = zeros(size(img));

% pad test image with 7 layers to the right and bottom
img_pad = img(:,:);
img_pad(end+1:end+7,:) = img(end-7:end-1,:);
img_pad(1:end-7,end+1:end+7) = img(:,end-7:end-1);
img_pad(end-7:end,end-7:end) = img(end-7:end,end-7:end);

PYGX_C_x = zeros(size(img));
PYGX_G_x = zeros(size(img));

img_feature_map = zeros(size(img));

for i=1:img_height
    for j=1:img_width
        img_block_dct = dct2(img_pad(i:i+7,j:j+7));
        block_dct_vec(1,zigzag_pat_lin) = img_block_dct(:);

        block_dct_feature = find_feature(block_dct_vec);
        img_feature_map(i,j) = block_dct_feature;

        PYGX_C_x(i,j) = (PXGY_x_C(block_dct_feature)*PY_C)/PX_x(block_dct_feature);
        PYGX_G_x(i,j) = (PXGY_x_G(block_dct_feature)*PY_G)/PX_x(block_dct_feature);

        if (PYGX_C_x(i,j) >= PYGX_G_x(i,j))
            seg_mask_res(i,j) = 1;
        else
            seg_mask_res(i,j) = 0;
        end
    end
end

figure;
FG_img_prob = imagesc(PYGX_C_x);
colormap(gray(255));
% imwrite(FG_img_prob, "../plots/FG_colormap.tiff");

figure;
BG_img_prob = imagesc(PYGX_G_x);
colormap(gray(255));
% imwrite(BG_img_prob, "../plots/BG_colormap.tiff");

% figure;
% imshow(img_pad);

figure;
imshow(seg_mask_res);
% imwrite(seg_mask_res, "../plots/seg_mask_res.png");

% load the ground-truth segmentation mask
seg_mask_gt = im2double(imread('../data/cheetah_mask.bmp'));
% imwrite(seg_mask_gt, '../plots/seg_mask_gt.png');
figure;
imshow(seg_mask_gt);

% estimate probability error using segmentation result and ground-truth
num_pixels = size(seg_mask_gt,1)*size(seg_mask_gt,2);
num_corr_pred = sum(seg_mask_gt == seg_mask_res, 'all');
num_incorr_pred = sum(seg_mask_gt ~= seg_mask_res, 'all');
error_frac = num_incorr_pred/num_pixels;

% estimate the minimum prob. of error
final_prob_error = 0;
for i=1:64
    num_pixels = sum(img_feature_map(:)==i);  % num pixels with feature=i
    if num_pixels ~= 0
        num_wrong_pred = sum(abs(seg_mask_res(img_feature_map==i) - seg_mask_gt(img_feature_map==i)));
        final_prob_error = final_prob_error + (num_wrong_pred/num_pixels)*PX_x(i);
    end
end