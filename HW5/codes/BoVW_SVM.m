clc
clear

%% Parameter

% Number of Clusters of kMeans
numClusters = 1000;

% sift_method = 1 for dsift
% sift_method = 2 for sift
sift_method = 2;

% Number of classes
catNum = 15;

% Parameter of vl_svmtrain LAMBDA
lambda = 5e-4;

% calculate_train = 0 for only testing data
% calculate_train = 1 for both training and testing data
calculate_train = 1;

%% Load data
[trainPaths, trainLabels] = LoadCSV("../csv/train.csv");
[testPaths, testLabels] = LoadCSV("../csv/test.csv");

trainImgs = getImgs(trainPaths);
testImgs = getImgs(testPaths);
trainNum = length(trainLabels);
testNum = length(testLabels);

%% Feature K-means
clc

if sift_method == 1
    sift = 'dsift';
elseif sift_method == 2
    sift = 'sift';
end
sift_features = [];
for i = 1:trainNum
    step_size = 20;
    img = reshape(trainImgs(i,:),256,256);
    if sift_method == 1
        [locations, sift_features_per_image] = vl_dsift(img, 'step', step_size);
    elseif sift_method == 2
        % Each column of d is the descriptor of one interest point in image I.
        [locations, sift_features_per_image] = vl_sift(img, 'PeakThresh', 10, 'EdgeThresh', 3) ;
    end
    sift_features = [sift_features, single(sift_features_per_image)];
end
[vocab, assignments] = vl_kmeans(sift_features, numClusters);

%% Training Histogram
clc
train_image_feats = BoVW(trainImgs, vocab, numClusters, trainNum);  % Training Histogram
test_image_feats = BoVW(testImgs, vocab, numClusters, testNum);  % Testing Histogram

%% SVM
clc

allMethod_accs = [];
train_accs = [];
test_accs = [];
xs = [];

if calculate_train == 1
    preds = SVM(train_image_feats, train_image_feats, trainNum, trainNum, trainLabels, catNum, numClusters, lambda);
    acc = getAcc(preds, trainLabels, trainNum);
    train_accs = [train_accs; acc];
end

preds = SVM(train_image_feats, test_image_feats, trainNum, testNum, trainLabels, catNum, numClusters, lambda);
acc = getAcc(preds, testLabels, testNum);
test_accs = [test_accs; acc];

% xs = [xs; numClusters];    
% f = figure('visible','off');
% plot(xs,test_accs,'-b','Linewidth',1.4);
% if calculate_train == 1
%     hold on
%     plot(xs,train_accs,'-r','Linewidth',1.4);
%     legend({'Test Acc', 'Train Acc'}, 'Location', 'southeast');
%     figPath = '../results/withTrainRes/';
% else
%     legend({'Test Acc'}, 'Location', 'southeast');
%     figPath = '../results/withoutTrainRes/';
% end    
% figName = strcat('BoVW-SVM (', sift, ')');
% title(figName);
% xlabel('# Clusters');
% ylabel('Accuracy');
% saveas(f, strcat(figPath, figName, '.png'));

test_accs = ['Test Acc'; convertCharsToStrings(sift); test_accs(1:end)];
if calculate_train == 1
    train_accs = ['Train Acc'; convertCharsToStrings(sift); train_accs(1:end)];
    allMethod_accs = [allMethod_accs, train_accs, test_accs];
else
    allMethod_accs = [allMethod_accs, test_accs];
end

%% Functions

function acc = getAcc(preds, gts, num)
    count = 0;
    for i = 1:num
       if preds(i) ==  gts(i)
           count = count + 1;
       end
    end
    acc = count / num;
end

function imgs = getImgs(imgPaths)
    imgs = [];
    for i = 1:length(imgPaths)
        path = imgPaths(i);
        img = imread(path);
        img = single(img);
        img = imresize(img, [256,256]);
        img = reshape(img,1,[]);
        imgs = [imgs; img];
    end
end