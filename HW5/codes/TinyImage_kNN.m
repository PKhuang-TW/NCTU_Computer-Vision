clc
clear

%% Parameters

% p = 1 for our own L1-norm from scratch
% p = 2 for our own L2-norm from scratch
% p = 3 for vlfeat --> vl_L1-norm
% p = 4 for vlfeat --> vl_L2-norm
p = 3;

% Range of k in kNN
mink = 1;  % min
maxk = 100;  % max

% myKNN_Bool = 0 for Matlab kNN
% myKNN_Bool = 1 for my kNN
myKNN_Bool = 1;

% calculate_train = 0 for only testing data
% calculate_train = 1 for both training and testing data
calculate_train = 1;

%% Load data
[trainPaths, trainLabels] = LoadCSV("../csv/train.csv");
[testPaths, testLabels] = LoadCSV("../csv/test.csv");

trainTinyImgs = getTinyImgs(trainPaths);
testTinyImgs = getTinyImgs(testPaths);
trainNum = length(trainLabels);
testNum = length(testLabels);

%% KNN algorithm
trainTinyImgs = double(trainTinyImgs);
testTinyImgs = double(testTinyImgs);

allMethod_accs = [];
if p == 1
    method = 'L1-norm';
elseif p == 2
    method = 'L2-norm';
elseif p == 3
    method = 'vl-L1-norm';
elseif p == 4
    method = 'vl-L2-norm';
end

xs = [];
train_accs = [];
test_accs = [];
for k = mink:1:maxk
    text = strcat('Predicting k=', int2str(k), '...');
    disp(text);
    if myKNN_Bool == 1  % My KNN
        if calculate_train == 1
            preds = myKNN(k, p, trainTinyImgs, trainTinyImgs, trainLabels, trainLabels);
            acc = getAcc(preds, trainLabels, trainNum);
            train_accs = [train_accs; acc];
        end

        preds = myKNN(k, p, trainTinyImgs, testTinyImgs, trainLabels, testLabels);
        acc = getAcc(preds, testLabels, testNum);
        test_accs = [test_accs; acc];

    elseif myKNN_Bool == 2  % Matlab KNN
        mdl = fitcknn(trainTinyImgs,trainLabels,'NumNeighbors',k);

        if calculate_train == 1
            [predLabels,score,cost] = predict(mdl,trainTinyImgs);
            acc = getAcc(predLabels, trainLabels, trainNum);
            train_accs = [train_accs; acc];
        end

        [predLabels, score, cost] = predict(mdl,testTinyImgs);
        acc = getAcc(predLabels, testLabels, testNum);
        test_accs = [test_accs; acc];
    end
    xs = [xs; k];
end
text = '--- KNN Done! ---';
disp(text);

% f = figure('visible','off');
% plot(xs,test_accs,'-b','Linewidth',1.4);
% if calculate_train == 1
%     hold on
%     plot(xs,train_accs,'-r','Linewidth',1.4);
%     legend({'Test Acc','Train Acc'},'Location','northeast');
% else
%     legend({'Test Acc'},'Location','northeast');
% end
% 
% if calculate_train == 1
%     figPath = '../results/withTrainRes/';
% else
%     figPath = '../results/withoutTrainRes/';
% end
% figName = strcat('TinyKNN (', method, ')');
% title(figName);
% xlabel('k');
% ylabel('Accuracy');
% saveas(f, strcat(figPath, figName, '.png'));

test_accs = ['Test Acc'; convertCharsToStrings(method); test_accs(1:end)];
if calculate_train == 1
    train_accs = ['Train Acc'; convertCharsToStrings(method); train_accs(1:end)];
    allMethod_accs = [allMethod_accs, train_accs, test_accs];
else
    allMethod_accs = [allMethod_accs, test_accs];
end


%% Function
function acc = getAcc(preds, gts, num)
    count = 0;
    for i = 1:num
       if preds(i) ==  gts(i)
           count = count + 1;
       end
    end
    acc = count / num;
end

function tinyImgs = getTinyImgs(imgPaths)
    tinyImgs = [];
    for i = 1:length(imgPaths)
        path = imgPaths(i);
        img = imread(path);
        img = imresize(img, [16,16]);
        img = reshape(img,1,[]);
        tinyImgs = [tinyImgs; img];
    end
end