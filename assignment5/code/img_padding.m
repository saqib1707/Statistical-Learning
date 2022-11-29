function [img_pad] = img_padding(img)
    [h, w] = size(img);
%     img_pad = zeros(h + 7, w + 7);
%     img_pad(4:h+3, 4:w+3) = img;
% 
%     img_pad(1:3, 4:w+3) = img(1:3, :);
%     img_pad(h+4:end, 4:w+3) = img(h-3:h, :);
%     img_pad(4:h+3, 1:3) = img(:, 1:3);
%     img_pad(4:h+3, w+2:end) = img(:, w-7:w-2);
%     img_pad(1:3, 1:3) = img(1:3, 1:3);

    img_pad = padarray(img, [3 3], 'replicate', 'pre');
    img_pad = padarray(img_pad, [4 4], 'replicate', 'post');
    
%     img_pad = img(:,:);
%     img_pad(end+1:end+7,:) = img(end-7:end-1,:);
%     img_pad(1:end-7,end+1:end+7) = img(:,end-7:end-1);
%     img_pad(end-7:end,end-7:end) = img(end-7:end,end-7:end);
end