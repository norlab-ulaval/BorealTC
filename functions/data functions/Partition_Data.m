function [Train,Test] = Partition_Data(REC,Channels,KFOLD,PART_WINDOW,RNG)

% FUNCTION OVERVIEW
%{
This function partitions the data contained in the struct "REC" for K-fold
cross validation according to the number of folders indicated in "KFOLD".
Uses the Data provided at higher frequency as reference to window the data
in windows of length "PART_WINDOW".
Finds the data provided at higher frequency according to the struct
"Channels".
Uses the struct "RNG" to govern the random process of partitioning.

Returns two structs "Train" and "Test" both with as many folders as
indicated by the parameter "KFOLD".
Each field of Train or Test appears as e.g. "Train.folder_n" where "n" goes
from 1 to KFOLD.
Each field "folder_n" conatins 3 fields:
 - data, containing a cell with all the data of same size 
 - time, containing the corresponding timestamps
 - labl, containing a categorical label corresponding to the terrain

The categorical labels follow the order of the fieldnames of the struct
"REC.data", so the category "0" corresponds to the first fieldname of
"REC.data" (or "REC.time" as well).
%}

% find the channel "c" providing data at higher frequency "sf"
CN = fieldnames(Channels);
c = 1;
sf = Channels.(CN{c}).sf;
for i=2:numel(CN)
    if Channels.(CN{i}).sf > sf
        sf = Channels.(CN{i}).sf;
        c = i;
    end
end

% window the data in a new struct "PRT" using "PARTITION_WINDOW" and the
% channel "c" as reference

TN = fieldnames(REC.data);
NumTer = numel(TN);
for i = 1:NumTer
    k = 1;
    for j = 1:size(REC.data.(TN{i}),1)
        strt = 1;
        stop = floor(PART_WINDOW*sf);
        while stop <= size(REC.data.(TN{i}){j,c},1)
            t0 = REC.time.(TN{i}){j,c}(strt);
            t1 = REC.time.(TN{i}){j,c}(stop);
            PRT.time.(TN{i}){k,c} = REC.time.(TN{i}){j,c}(strt:stop);
            PRT.data.(TN{i}){k,c} = REC.data.(TN{i}){j,c}(strt:stop,:);
            for s = 1:numel(CN)
                if s~=c
                    [~,e0] = min(abs(t0-REC.time.(TN{i}){j,s}));
                    [~,e1] = min(abs(t1-REC.time.(TN{i}){j,s}));
                    PRT.time.(TN{i}){k,s} = REC.time.(TN{i}){j,s}(e0:e1);
                    PRT.data.(TN{i}){k,s} = REC.data.(TN{i}){j,s}(e0:e1,:);
                end
            end
            strt = stop+1;
            stop = stop+floor(PART_WINDOW*sf);
            k = k+1;
        end
    end
end
% the user can uncomment the following line to visualize the "PRT" struct
% assignin('base','PRT',PRT);

% create a struct "UNF" containing all "PRT" data and time
% create a categorical array "label" for partitioning

k = 1;
for i = 1:NumTer
    for j = 1:size(PRT.data.(TN{i}),1)
        UNF.data(k,:) = PRT.data.(TN{i})(j,:);
        UNF.time(k,:) = PRT.time.(TN{i})(j,:);
        labels(k,1) = categorical(i-1);
        k = k+1;
    end
end
% the user can uncomment the following line to visualize the "UNF" struct
% assignin('base','UNF',UNF);

% use cvpartition on "labels" categorical array to partition data
rng(RNG.seed,RNG.generator)
cvp = cvpartition(labels,'KFold',KFOLD,'Stratify',true);

% create "Train" and "Test" structs
for i = 1:KFOLD
    Trn_Ind = find(training(cvp,i));
    for j = 1:length(Trn_Ind)
        Train.(strcat('folder_',num2str(i))).data(j,:) = UNF.data(Trn_Ind(j),:);
        Train.(strcat('folder_',num2str(i))).time(j,:) = UNF.time(Trn_Ind(j),:);
        Train.(strcat('folder_',num2str(i))).labl(j,1) = labels(Trn_Ind(j));
    end
    Tst_Ind = find(test(cvp,i));
    for j = 1:length(Tst_Ind)
        Test.(strcat('folder_',num2str(i))).data(j,:) = UNF.data(Tst_Ind(j),:);
        Test.(strcat('folder_',num2str(i))).time(j,:) = UNF.time(Tst_Ind(j),:);
        Test.(strcat('folder_',num2str(i))).labl(j,1) = labels(Tst_Ind(j));
    end
end

end