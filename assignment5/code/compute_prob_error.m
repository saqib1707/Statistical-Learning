function [prob_error] = compute_prob_error(seg_mask_gt, seg_mask_res, PY_FG, PY_BG, method)
    [h, w] = size(seg_mask_gt);

    if (method == 1)
        % Method-1 of computing probability of error
        num_FG_pixels = sum(seg_mask_gt, 'all');
        num_BG_pixels = h * w - num_FG_pixels;
    
        count_FG_pixels = 0;
        count_BG_pixels = 0;
        count_FG_error = 0;
        count_BG_error = 0;
    
        for i = 1:h 
            for j = 1:w 
                if (seg_mask_gt(i,j) == 1)
                    count_FG_pixels = count_FG_pixels + 1;
                    if (seg_mask_res(i,j) == 0)
                        count_FG_error = count_FG_error + 1;
                    end
                elseif (seg_mask_gt(i,j) == 0)
                    count_BG_pixels = count_BG_pixels + 1;
                    if (seg_mask_res(i,j) == 1)
                        count_BG_error = count_BG_error + 1;
                    end
                end
            end
        end
    
        assert(num_FG_pixels == count_FG_pixels);
        assert(num_BG_pixels == count_BG_pixels);
    
        prob_error = (count_FG_error / count_FG_pixels) * PY_FG + (count_BG_error / count_BG_pixels) * PY_BG;
    elseif (method == 2)
        % Method-2 of computing probability of error
        num_error_pixels = sum(abs(seg_mask_gt - seg_mask_res), 'all');
        num_pixels = h * w;
        prob_error = num_error_pixels / num_pixels;
    end
end