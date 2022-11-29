function img_feature = find_feature(img_dct)
    [num_rows,~] = size(img_dct);
    % disp(num_rows);
    img_feature = zeros(num_rows,1);
    img_dct_abs = abs(img_dct);
    
    % convert 64-D feature to 1-D feature using 2nd largest energy
    for i=1:num_rows
        [~,max_idx] = max(img_dct_abs(i,:));    % 1st largest coefficient
        img_dct_abs(i,max_idx) = -Inf;
        [~,max_idx] = max(img_dct_abs(i,:));    % 2nd largest coefficient

        img_feature(i,1) = max_idx;
    end
end