function [TrainMCS,TestMCS] = MCS_Data(Train,Test,Channels,CNNpar)

% FUNCTION OVERVIEW
%{
This function transforms the partitioned data contained in the structs
"Train" and "Test" into the corresponding multichannel spectrograms
contained in the structs "TrainMCS" and "TestMCS".
The construction of the spectrograms is done accordingly to the time window
specified in the input "CNNpar.TimeWindow" and time overlap
"CNNpar.TimeOvrlap". 
The contruction of the spectrograms uses the information contained in the
struct "Channels" passed to the function "Multichannel_Spectrogram"

The function "Multichannel_Spectrogram" can be found in the directory
"./functions/spectrogram functions" with a function overview that the user
can read to better understand the process of multichannel spectrograms
construction.
%}

tw = CNNpar.TimeWindow;
to = CNNpar.TimeOvrlap;

FN = fieldnames(Train);
Kfold = numel(FN);

for i = 1:Kfold
    for j = 1:size(Train.(FN{i}).data,1)
    [TrainMCS.(FN{i}).data{j,1},TrainMCS.(FN{i}).time(j,:),TrainMCS.(FN{i}).freq(j,:)] = ...
        Multichannel_Spectrogram(Train.(FN{i}).data(j,:),Train.(FN{i}).time(j,:),Channels,tw,to);
    end
    TrainMCS.(FN{i}).labl = Train.(FN{i}).labl;
    
    for j = 1:size(Test.(FN{i}).data,1)
    [TestMCS.(FN{i}).data{j,1},TestMCS.(FN{i}).time(j,:),TestMCS.(FN{i}).freq(j,:)] = ...
        Multichannel_Spectrogram(Test.(FN{i}).data(j,:),Test.(FN{i}).time(j,:),Channels,tw,to);
    end
    TestMCS.(FN{i}).labl = Test.(FN{i}).labl;
end

end