function RES = LSTM_RecurrentNet(TrainDS,TestDS,LSTMpar,LSTM_TrainOpt,RNG)

% FUNCTION OVERVIEW
%{
This functions performs training and testing of a Long Short-Term Memory
Recurrent Neural Network model.
This model takes as input the downsampled data sequences contained in the
structs "TrainDS" and "TestDS".
The function returns a struct "RES" containing as many fields as the k-fold
cross validation used to partition the data in "TrainDS" and "TestDS",
plus one extrafield with the resulting confusion matrix of the entire
k-fold cross validation process.
Each field of "RES" corresponding to cross validation folders contains 4
fields:
    - TrainingTime, with the time in milliseconds required to train the
    model corresponding to the folder
    - TestingTime, with the time in milliseconds required to test the
    model corresponding to the folder
    - Model, with the Long Short-Term Memory Recurrent Neural Network
    resulting from the training process performed on the corresponding
    folder 
    - ConfusionMat, containing the confusion matrix resulting from the
    testing of the model on the correspoding folder

The architecture of the net is defined by the array "layer" and the
lstmLayer is construced according to the parameter contained in the input
struct "LSTMpar". 

The training options are defined in the object "opt" accordingly to the
input cell "LSTM_TrainOpt".

The validation set is constructed using the matlab function "cvpartition"
and the random process is governed by the input struct "RNG".

The partitioned dataset contained in the input structs "TrainDS" and
"TestDS" is at first rearranged in "XTrain", "XValid", "XTest", "YTrain",
"YValid" and "YTest" then passed to the matlab function "trainNetwork" to
perform the training process.

Testing of the model is performed using the matlab function "classify".
%}

% assign the training options to appropriate variables
for i = 1:size(LSTM_TrainOpt,1)
    switch LSTM_TrainOpt{i,1}
        case 'valid_perc'
            valid_perc = LSTM_TrainOpt{i,2};
        case 'init_learn_rate'
            init_learn_rate = LSTM_TrainOpt{i,2};
        case 'learn_drop_factor'
            learn_drop_factor = LSTM_TrainOpt{i,2};
        case 'max_epochs'
            max_epochs = LSTM_TrainOpt{i,2};
        case 'minibatch_size'
            minibatch_size = LSTM_TrainOpt{i,2};
        case 'valid_patience'
            valid_patience = LSTM_TrainOpt{i,2};
        case 'valid_frequency'
            valid_frequency = LSTM_TrainOpt{i,2};
        case 'gradient_treshold'
            gradient_treshold = LSTM_TrainOpt{i,2};
    end
end

FN = fieldnames(TrainDS);
Kfold = numel(FN);

% define the architecture of LSTM
InSize = size(TrainDS.(FN{1}).data{1},2);
TotLabl = [TrainDS.(FN{1}).labl;TestDS.(FN{1}).labl];
NumTer = max(double(TotLabl));

layers = [ ...
    sequenceInputLayer(InSize,"Name","sequenceinput")
    lstmLayer(LSTMpar.nHiddenUnits,'OutputMode','last',"Name","lstm")
    fullyConnectedLayer(NumTer,"Name","fullyConn")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classLayer")];

% find validation indexes
iValid = cell(Kfold,1);
for i = 1:Kfold
    rng(RNG.seed,RNG.generator)
    cvp = cvpartition(size(TrainDS.(FN{i}).labl,1),'HoldOut',valid_perc);
    iValid{i} = cvp.test;
end

% arrange the data how the training function requires them, train and test
% the model

for i = 1:Kfold
    XTrain = cellfun(@transpose,TrainDS.(FN{i}).data(~iValid{i}),'un',0);
    YTrain = TrainDS.(FN{i}).labl(~iValid{i});
    
    XValid = cellfun(@transpose,TrainDS.(FN{i}).data(iValid{i}),'un',0);
    YValid = TrainDS.(FN{i}).labl(iValid{i});
    
    XTest = cellfun(@transpose,TestDS.(FN{i}).data,'un',0);
    YTest = TestDS.(FN{i}).labl;
    
    % define training options
    opt = trainingOptions('adam', ...
            'ExecutionEnvironment','cpu', ...
            'GradientThreshold',gradient_treshold, ...
            'InitialLearnRate',init_learn_rate, ...
            'LearnRateSchedule','piecewise', ...
            'Verbose',0,...
            'LearnRateDropFactor',learn_drop_factor, ...
            'MaxEpochs',max_epochs, ...
            'MiniBatchSize',minibatch_size, ...
            'ValidationData',{XValid,YValid}, ...
            'ValidationPatience', valid_patience, ...
            'ValidationFrequency',valid_frequency, ...   
            'SequenceLength','longest', ...
            'Shuffle','every-epoch', ...
            'Plots','training-progress');
    
    % Train LSTM
    disp(strcat('LSTM Training Partition'," ",num2str(i)))
    
    tic
    [LSTM,info] = trainNetwork(XTrain,YTrain,layers,opt);
    % store results
    RES.(FN{i}).TrainingTime = toc;
    RES.(FN{i}).Model = LSTM;
    
    MaxIter = floor(size(XTrain,1)/minibatch_size)*max_epochs;
    VldLoss = info.ValidationLoss(~isnan(info.ValidationLoss));
    [~,minVldLossInd] = min(VldLoss);
    if numel(VldLoss(minVldLossInd+1:end)) >= valid_patience
        disp('Model training complete: met validation criterion')
        disp('Final training accuracy:')
        disp(strcat(num2str(info.TrainingAccuracy(end)),'%'))
        disp('Final validation accuracy:')
        disp(strcat(num2str(info.FinalValidationAccuracy),'%'))
    elseif numel(info.TrainingLoss) >= MaxIter
        disp('Model training complete: reached final iteration')
        disp('Final training accuracy:')
        disp(strcat(num2str(info.TrainingAccuracy(end)),'%'))
        disp('Final validation accuracy:')
        disp(strcat(num2str(info.FinalValidationAccuracy),'%'))
    end
    disp('--------------------------------------------------')
    % Test LSTM
    tic
    YPred = classify(LSTM,XTest);
    % store results
    RES.(FN{i}).TestingTime = toc;
    RES.(FN{i}).ConfusionMat = confusionmat(YTest,YPred);
    
    % clear variables for the next folder
    clear XTrain XValid XTest
    clear YTrain YValid YTest
    clear YPred
    delete(findall(0));
    
end
disp('--------------------------------------------------')
% compute the confusion matrix of the k-fold cross validation process
CM = RES.(FN{1}).ConfusionMat;
for i = 2:Kfold
    CM = CM + RES.(FN{i}).ConfusionMat;
end

% store the results
RES.ConfusionMat = CM;


end