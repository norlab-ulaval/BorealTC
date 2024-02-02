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
channelNames = fieldnames(Channels);
c = 1;
sf = Channels.(channelNames{c}).sf;
for i=2:numel(channelNames)
    if Channels.(channelNames{i}).sf > sf
        sf = Channels.(channelNames{i}).sf;
        c = i;
    end
end

FN = fieldnames(Train);
Kfold = numel(FN);
hf = Channels.(channelNames{c}).sf;
MW = floor(w * hf);

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
        stop = strt + MW - 1;
        while stop < size(Samp,1)
            GenSmp{k,1} = Samp(strt:stop,:);
            strt = strt+(AUG.sliding_window*hf);
            stop = strt + MW - 1;
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
            TerSli(i) = (1/hf)*((size(Samp,1)-MW)/TerSmp(i));
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

clear s

for i = 1:Kfold
    fold = FN{i};
    k = 1;
    numPartitions = size(Train.(fold).data,1);
    for partIdx = 1:numPartitions
        sli = TerSli(double(Train.(fold).labl(partIdx)));
        strt = 1;
        stop = strt + MW - 1;
        while stop <= size(Train.(fold).data{partIdx,c},1)
            AugTrain.(fold).data{k,c} = Train.(fold).data{partIdx,c}(strt:stop,:);
            AugTrain.(fold).time{k,c} = Train.(fold).time{partIdx,c}(strt:stop);
            AugTrain.(fold).labl(k,1) = Train.(fold).labl(partIdx);
            t0 = Train.(fold).time{partIdx,c}(strt);
            t1 = Train.(fold).time{partIdx,c}(stop);
            for chanIdx = 1:numel(channelNames)
                if chanIdx~=c
                    [~,e0] = min(abs(t0-Train.(fold).time{partIdx,chanIdx}));
                    [~,e1] = min(abs(t1-Train.(fold).time{partIdx,chanIdx}));
                    AugTrain.(fold).data{k,chanIdx} = Train.(fold).data{partIdx,chanIdx}(e0:e1,:);
                    AugTrain.(fold).time{k,chanIdx} = Train.(fold).time{partIdx,chanIdx}(e0:e1);
                    % make the dimensions omogeneus
                    if size(AugTrain.(fold).data{k,chanIdx},1) > round(w*Channels.(channelNames{chanIdx}).sf)
                        AugTrain.(fold).data{k,chanIdx}(end,:) = [];
                        AugTrain.(fold).time{k,chanIdx}(end) = [];
                    end
                end
            end
            strt = strt + floor(sli*hf);
            stop = strt + MW - 1;
            k = k+1;
        end
    end

    k = 1;
    for partIdx = 1:size(Test.(fold).data,1)
        sli = TerSli(double(Test.(fold).labl(partIdx)));
        strt = 1;
        stop = strt + MW - 1;
        while stop <= size(Test.(fold).data{partIdx,c},1)
            AugTest.(fold).data{k,c} = Test.(fold).data{partIdx,c}(strt:stop,:);
            AugTest.(fold).time{k,c} = Test.(fold).time{partIdx,c}(strt:stop);
            AugTest.(fold).labl(k,1) = Test.(fold).labl(partIdx);
            t0 = Test.(fold).time{partIdx,c}(strt);
            t1 = Test.(fold).time{partIdx,c}(stop);
            for chanIdx = 1:numel(channelNames)
                if chanIdx~=c
                    [~,e0] = min(abs(t0-Test.(fold).time{partIdx,chanIdx}));
                    [~,e1] = min(abs(t1-Test.(fold).time{partIdx,chanIdx}));
                    AugTest.(fold).data{k,chanIdx} = Test.(fold).data{partIdx,chanIdx}(e0:e1,:);
                    AugTest.(fold).time{k,chanIdx} = Test.(fold).time{partIdx,chanIdx}(e0:e1);
                    % make the dimensions homogeneus
                    if size(AugTest.(fold).data{k,chanIdx},1) > round(w*Channels.(channelNames{chanIdx}).sf)
                        AugTest.(fold).data{k,chanIdx}(end,:) = [];
                        AugTest.(fold).time{k,chanIdx}(end) = [];
                    end
                end
            end
            strt = strt + floor(sli*hf);
            stop = strt + MW - 1;
            k = k+1;
        end
    end

end


end
