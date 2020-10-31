function preds = myKNN(k, p, trainImgs, testImgs, trainLabels, testLabels)
    preds = [];
    trainNum = length(trainLabels);
    testNum = length(testLabels);
    
    if p >= 3  % vl_alldist2
        % Row_j of dist is distance from testImg_j to every trainImgs
        if p == 3
            dists = vl_alldist2(testImgs', trainImgs', 'L1');
        elseif p == 4
            dists = vl_alldist2(testImgs', trainImgs', 'L2');
        end
        
        for idx = 1:testNum
            dist = dists(idx, :);
            label = findLabel(trainLabels, dist, k);
            preds = [preds; label];  % mode finds the most frequent value in the array
        end
    else
        for idx = 1:testNum

            %Step 1: Computing distance for each testdata
            R = repmat(testImgs(idx,:), trainNum, 1);
            
            if p == 1  % L1-norm
                dist = abs(R - (trainImgs));
            elseif p == 2  % L2-norm
                dist = (R - trainImgs).^2;
            end
            dist = sum(dist, 2);
            
            label = findLabel(trainLabels, dist, k);
            preds = [preds; label];  % mode finds the most frequent value in the array
        end
    end
    
   
    function mostFreqLabel = findLabel(trainLabels, dist, k)
        labs = [];
         %Step 2: compute k nearest neighbors and store them in an array
        [d_, order] = sort(dist);
        KNNs = order(1:k);
        KNDist = d_(1:k);

        % Step 3 : Voting 
        for i=1:k
            labs = [labs; trainLabels(KNNs(i))];
        end
        mostFreqLabel = mode(labs);