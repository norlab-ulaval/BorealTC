function RES = SupportVectorMachine(Train,Test,SVMpar,SVM_TrainOpt)

% FUNCTION OVERVIEW
%{

%}

% assign the training options to appropriate variables
for i = 1:size(SVM_TrainOpt,1)
    switch SVM_TrainOpt{i,1}
        case 'kernel_function'
            kernel_function = SVM_TrainOpt{i,2};
        case 'polynomial_order'
            polynomial_order = SVM_TrainOpt{i,2};
        case 'kernel_scale'
            kernel_scale = SVM_TrainOpt{i,2};
        case 'box_constraint'
            box_constraint = SVM_TrainOpt{i,2};
        case 'standardize'
            standardize = SVM_TrainOpt{i,2};
        case 'coding'
            coding = SVM_TrainOpt{i,2};
    end
end

FN = fieldnames(Train);
Kfold = numel(FN);

% find the sizes of every channel and save them in the array "sz"
sz = zeros(1,size(Train.(FN{1}).data,2));
for c = 1:size(Train.(FN{1}).data,2)
    sz(c) = size(Train.(FN{1}).data{1,c},2);
end

% arrange the data how the training function requires them, train and test
% the model

for i = 1:Kfold
    
    for m = 1:SVMpar.nStatMom
        switch m
            case 1
                for j = 1:size(Train.(FN{i}).data,1)
                    XTrain(j,1:sz(1)) = mean(Train.(FN{i}).data{j,1});
                    for c = 2:numel(sz)
                        XTrain(j,sum(sz(1:c-1))+1:sum(sz(1:c-1))+sz(c)) = mean(Train.(FN{i}).data{j,c});
                    end
                end
            case 2
                for j = 1:size(Train.(FN{i}).data,1)
                    XTrain(j,sum(sz)+1:sum(sz)+sz(1)) = std(Train.(FN{i}).data{j,1});
                    for c = 2:numel(sz)
                        XTrain(j,sum(sz)+sum(sz(1:c-1))+1:sum(sz)+sum(sz(1:c-1))+sz(c)) = std(Train.(FN{i}).data{j,c});
                    end
                end
            case 3
                for j = 1:size(Train.(FN{i}).data,1)
                    XTrain(j,2*sum(sz)+1:2*sum(sz)+sz(1)) = skewness(Train.(FN{i}).data{j,1});
                    for c = 2:numel(sz)
                        XTrain(j,2*sum(sz)+sum(sz(1:c-1))+1:2*sum(sz)+sum(sz(1:c-1))+sz(c)) = skewness(Train.(FN{i}).data{j,c});
                    end
                end
            case 4
                for j = 1:size(Train.(FN{i}).data,1)
                    XTrain(j,3*sum(sz)+1:3*sum(sz)+sz(1)) = kurtosis(Train.(FN{i}).data{j,1});
                    for c = 2:numel(sz)
                        XTrain(j,3*sum(sz)+sum(sz(1:c-1))+1:3*sum(sz)+sum(sz(1:c-1))+sz(c)) = kurtosis(Train.(FN{i}).data{j,c});
                    end
                end
        end
    end
    YTrain = Train.(FN{i}).labl;
    
    for m = 1:SVMpar.nStatMom
        switch m
            case 1
                for j = 1:size(Test.(FN{i}).data,1)
                    XTest(j,1:sz(1)) = mean(Test.(FN{i}).data{j,1});
                    for c = 2:numel(sz)
                        XTest(j,sum(sz(1:c-1))+1:sum(sz(1:c-1))+sz(c)) = mean(Test.(FN{i}).data{j,c});
                    end
                end
            case 2
                for j = 1:size(Test.(FN{i}).data,1)
                    XTest(j,sum(sz)+1:sum(sz)+sz(1)) = std(Test.(FN{i}).data{j,1});
                    for c = 2:numel(sz)
                        XTest(j,sum(sz)+sum(sz(1:c-1))+1:sum(sz)+sum(sz(1:c-1))+sz(c)) = std(Test.(FN{i}).data{j,c});
                    end
                end
            case 3
                for j = 1:size(Test.(FN{i}).data,1)
                    XTest(j,2*sum(sz)+1:2*sum(sz)+sz(1)) = skewness(Test.(FN{i}).data{j,1});
                    for c = 2:numel(sz)
                        XTest(j,2*sum(sz)+sum(sz(1:c-1))+1:2*sum(sz)+sum(sz(1:c-1))+sz(c)) = skewness(Test.(FN{i}).data{j,c});
                    end
                end
            case 4
                for j = 1:size(Test.(FN{i}).data,1)
                    XTest(j,3*sum(sz)+1:3*sum(sz)+sz(1)) = kurtosis(Test.(FN{i}).data{j,1});
                    for c = 2:numel(sz)
                        XTest(j,3*sum(sz)+sum(sz(1:c-1))+1:3*sum(sz)+sum(sz(1:c-1))+sz(c)) = kurtosis(Test.(FN{i}).data{j,c});
                    end
                end
        end
    end
    YTest = Test.(FN{i}).labl;
    
    learner = templateSVM('KernelFunction', kernel_function,...
            'PolynomialOrder', polynomial_order,...
            'KernelScale', kernel_scale, ...
            'BoxConstraint', box_constraint, ...
            'Standardize', standardize);
    
    % Training SVM
    disp(strcat('SVM Training ',' Partition = ',num2str(i)))
    tic
    SVM = fitcecoc(XTrain,YTrain,'Learners',learner,'Coding',coding);
    RES.(FN{i}).TrainingTime = toc;
    RES.(FN{i}).Model = SVM;
    
    % Test SVM
    tic
    YPred = predict(SVM,XTest);
    RES.(FN{i}).TestingTime = toc;
    RES.(FN{i}).ConfusionMat = confusionmat(YTest,YPred);
    
    clear XTrain XValid XTest
    clear YTrain YValid YTest
    clear YPred
end

% compute the confusion matrix of the k-fold cross validation process
CM = RES.(FN{1}).ConfusionMat;
for i = 2:Kfold
    CM = CM + RES.(FN{i}).ConfusionMat;
end

% store the results
RES.ConfusionMat = CM;


end