function [AugTrain,AugTest] = Augment_Data(Train,Test,Channels,w,AUG)

% FUNCTION OVERVIEW
%{
This function augments the prevoiusly partitioned dataset using the sliding
window in "AUG.sliding_window". If the dataset does not have an equal
number of partition samples for all the terrains the code can adjust the
sliding window among terrains to have the same number of augmented samples
among all terrains. This modality is activated setting AUG.same = 1.
This function uses as reference for augmentation procedure the channel
providing data at higher frequency. This information is stored in the
struct "Channels".
This function returns two structs:
 - AugTrain, containing the augmented data, timestamps and labels for
 training a model 
 - AugTest, containing the augmented data, timestamps and labels for
 testing a model
The augmented data consists in samples of time length indicated by the
double "w" given as input.
%}

% find the channel "c" providing data at higher frequency "sf" to be used
% as reference for windowing operation
CN = fieldnames(Channels);
c = 1;
sf = Channels.(CN{c}).sf;
for i=2:numel(CN)
    if Channels.(CN{i}).sf > sf
        sf = Channels.(CN{i}).sf;
        c = i;
    end
end

FN = fieldnames(Train);
Kfold = numel(FN);

% decide whether augment the data omogeneusly or not

switch AUG.same
    
    case 1
        % find out the available amount of data for all terrains and store it into the
        % array "SizTer"

        TotLabl = [Train.(FN{1}).labl;Test.(FN{1}).labl];
        NumTer = max(double(TotLabl));
        SizTer = zeros(NumTer,1);
        for i = 1:NumTer
            SizTer(i) = numel(find(double(TotLabl)==i));
        end

        % find out the terrain with the smallest amount of data available and store
        % the corresponding number of samples in "MinNum" 

        [MinNum,~] = min(SizTer);

        % find out how many samples are generated from the augmentation of one
        % partition window, store the information in "GenSmp"

        k = 1;
        Samp = Train.(FN{1}).data{1,c};
        strt = 1;
        stop = strt + floor(w*Channels.(CN{c}).sf) - 1;
        while stop < size(Samp,1)
            GenSmp{k,1} = Samp(strt:stop,:);
            strt = strt+(AUG.sliding_window*Channels.(CN{c}).sf);
            stop = strt + floor(w*Channels.(CN{c}).sf) - 1;
            k = k+1;
        end

        % find out how many augmented samples will be available for the terrain
        % with the smallest amount of data, store the information in "TotSmp"

        TotSmp = size(GenSmp,1)*MinNum;

        % find out how many augmented samples have to be generated from one
        % partition window for all the terrains in order to have the same number of
        % augmented samples available for all terrains. Store the information in
        % the array "TerSmp"

        TerSmp = zeros(NumTer,1);
        for i = 1:NumTer
            TerSmp(i) = floor(TotSmp/SizTer(i));
        end

        % find out the sliding window that will generate the right amount of
        % augmented samples for each terrain, store the information in the array
        % "TerSli" 

        TerSli = zeros(NumTer,1);
        for i = 1:NumTer
            TerSli(i) = (1/Channels.(CN{c}).sf)*((size(Samp,1)-floor(w*Channels.(CN{c}).sf))/TerSmp(i));
        end
        
    case 0
        TotLabl = [Train.(FN{1}).labl;Test.(FN{1}).labl];
        NumTer = max(double(TotLabl));
        TerSli = zeros(NumTer,1);
        for i = 1:NumTer
            TerSli(i) = AUG.sliding_window;
        end
end

% augment the data using the appropriate sliding window for different
% terrains or the same for every terrain depending on AUG.same

for i = 1:Kfold
    k = 1;
    for j = 1:size(Train.(FN{i}).data,1)
        sli = TerSli(double(Train.(FN{i}).labl(j)));
        strt = 1;
        stop = strt + floor(w*Channels.(CN{c}).sf) - 1;
        while stop <= size(Train.(FN{i}).data{j,c},1)
            AugTrain.(FN{i}).data{k,c} = Train.(FN{i}).data{j,c}(strt:stop,:);
            AugTrain.(FN{i}).time{k,c} = Train.(FN{i}).time{j,c}(strt:stop);
            AugTrain.(FN{i}).labl(k,1) = Train.(FN{i}).labl(j);
            t0 = Train.(FN{i}).time{j,c}(strt);
            t1 = Train.(FN{i}).time{j,c}(stop);
            for s = 1:numel(CN)
                if s~=c
                    [~,e0] = min(abs(t0-Train.(FN{i}).time{j,s}));
                    [~,e1] = min(abs(t1-Train.(FN{i}).time{j,s}));
                    AugTrain.(FN{i}).data{k,s} = Train.(FN{i}).data{j,s}(e0:e1,:);
                    AugTrain.(FN{i}).time{k,s} = Train.(FN{i}).time{j,s}(e0:e1);
                    % make the dimensions omogeneus
                    if size(AugTrain.(FN{i}).data{k,s},1) > round(w*Channels.(CN{s}).sf)
                        AugTrain.(FN{i}).data{k,s}(end,:) = [];
                        AugTrain.(FN{i}).time{k,s}(end) = [];
                    end
                end
            end
            strt = strt + floor(sli*Channels.(CN{c}).sf);
            stop = strt + floor(w*Channels.(CN{c}).sf) - 1;
            k = k+1;
        end
    end
    
    k = 1;
    for j = 1:size(Test.(FN{i}).data,1)
        sli = TerSli(double(Test.(FN{i}).labl(j)));
        strt = 1;
        stop = strt + floor(w*Channels.(CN{c}).sf) - 1;
        while stop <= size(Test.(FN{i}).data{j,c},1)
            AugTest.(FN{i}).data{k,c} = Test.(FN{i}).data{j,c}(strt:stop,:);
            AugTest.(FN{i}).time{k,c} = Test.(FN{i}).time{j,c}(strt:stop);
            AugTest.(FN{i}).labl(k,1) = Test.(FN{i}).labl(j);
            t0 = Test.(FN{i}).time{j,c}(strt);
            t1 = Test.(FN{i}).time{j,c}(stop);
            for s = 1:numel(CN)
                if s~=c
                    [~,e0] = min(abs(t0-Test.(FN{i}).time{j,s}));
                    [~,e1] = min(abs(t1-Test.(FN{i}).time{j,s}));
                    AugTest.(FN{i}).data{k,s} = Test.(FN{i}).data{j,s}(e0:e1,:);
                    AugTest.(FN{i}).time{k,s} = Test.(FN{i}).time{j,s}(e0:e1);
                    % make the dimensions omogeneus
                    if size(AugTest.(FN{i}).data{k,s},1) > round(w*Channels.(CN{s}).sf)
                        AugTest.(FN{i}).data{k,s}(end,:) = [];
                        AugTest.(FN{i}).time{k,s}(end) = [];
                    end
                end
            end
            strt = strt + floor(sli*Channels.(CN{c}).sf);
            stop = strt + floor(w*Channels.(CN{c}).sf) - 1;
            k = k+1;
        end
    end
    
end


end
