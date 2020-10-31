function img_feats = BoVW(imgs, vocabMu, numClusters, dataNum)
    step_size = 10;
    img_feats = zeros(dataNum, numClusters);
    for i = 1:dataNum
        img = reshape(imgs(i,:),256,256);
        [locations, sift_features_per_image] = vl_dsift(img, 'step', step_size);

        % Row_j of D is distance from feature_j to every vocab mean
        D = vl_alldist2(single(sift_features_per_image), vocabMu);
        
        % Build Histogram
        hist = zeros(1, numClusters);
        for j = 1:size(D, 1)
            D_of_image = D(j, :);
            [val, idx] = min(D_of_image);
            hist(1, idx) = hist(1, idx) + 1;
        end
        
%         % Normalize the histogram
%         hist_zero_mean = hist - mean(hist);
%         img_feats(i, :) = hist_zero_mean ./ norm(hist_zero_mean);
        
        % Non-normalize
        img_feats(i, :) = hist;
    end
    
    