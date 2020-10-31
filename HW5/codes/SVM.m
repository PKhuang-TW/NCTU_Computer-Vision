function preds = SVM(train_img_feats, test_img_feats, trainNum, testNum, trainLabels, catNum, numClusters, lambda)
   
    [W, B] = weightNbias(train_img_feats, trainLabels, trainNum, catNum, numClusters, lambda);
    preds = [];
    
    % compute the score and assign the category with the highest score
    for j = 1:testNum
        max_score = -1e16;
        max_cat = [];
        xtest = test_img_feats(j, :)';
        scores = zeros(1, catNum);
        for i = 1:catNum
            scores(1, i) = W(:, i)' * xtest + B(1, i);
            if(scores(1, i) > max_score)
                max_score = scores(1, i);
                max_cat = i;
            end
        end
        preds = [preds; max_cat];
    end

    % Calculate the weight and the bias of the hyperplane
    function [W, B] = weightNbias(train_img_feats, trainLabels, trainNum, catNum, numClusters, lambda)
        W = zeros(numClusters, catNum);
        B = zeros(1, catNum);
        for i = 1:catNum
            labels = [];
            for j = 1:trainNum
                if(trainLabels(j) == i)
                    labels = [labels, 1];
                else
                    labels = [labels, -1];
                end
            end
            [w, b] = vl_svmtrain(train_img_feats', labels, lambda);
            W(:, i) = w;
            B(1, i) = b;
        end