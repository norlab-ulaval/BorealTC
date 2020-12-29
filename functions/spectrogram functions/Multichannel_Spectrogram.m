function [MCS,TimeGrids,FreqGrids] = Multichannel_Spectrogram(SignalCell,TimeCell,Channels,tw,to)

% FUNCTION OVERVIEW
%{
This assemble a multichannel spectrogram in "MCS" given a signal cell as
input in "SignalCell" and the corresponding timestamps in the input cell
"TimeCell". 
"SignalCell" must have as many columns as fieldnames of the input struct
"Channels" corresponding to data provided at different sampling
frequencies.
Each element of "SignalCell" must be a matrix of dimension 
[ nObs , nSensChan ].
This function builds spectrograms according to time window specified in the
input double "tw" and time overlap specified in "to".
This function performs a padding procedure to join spectrograms provided at
different sampling frequencies.
This function returns also 2 cells:
    - TimeGrids, containing the time specification for each sampling
    frequency
    - FreqGrids, containing the frequency specification for each sampling
    frequency
%}

% Build Multichannel Spectrogrm from omogeneus signals and save them in a
% cell object "cell_MCS"

CN = fieldnames(Channels);
for c = 1:numel(CN)
    for i = 1:size(SignalCell{1,c},2)
        [cell_MCS{1,c}(:,:,i),~,TimeGrids{1,c},FreqGrids{1,c}] = Spectrogram(SignalCell{1,c}(:,i),TimeCell{1,c},Channels.(CN{c}).sf,tw,to);
    end
end
% Build an array "sfs" with the sampling frequencies of all channels

sfs = zeros(numel(CN),1);
for c = 1:numel(CN)
    sfs(c) = Channels.(CN{c}).sf;
end

[hsf,hc] = max(sfs); % channel with higher sampling frequency

% Join Multichannel unomogeneus spectrograms in the same multichannel
% spectrogram "MCS" by padding the channels with lower sampling frequencies

for c = 1:numel(CN)
    if Channels.(CN{c}).sf ~= hsf
        sz = size(cell_MCS{1,hc});
        for i = 1:size(cell_MCS{1,c},3)
            pad_MCS{1,c}(:,:,i) = matrix_padding(cell_MCS{1,c}(:,:,i),sz(1:2));
        end
        TimeGrids{1,c} = matrix_padding(TimeGrids{1,c},sz(1:2));
        FreqGrids{1,c} = matrix_padding(FreqGrids{1,c},sz(1:2));
    else
        pad_MCS{1,c} = cell_MCS{1,hc};
    end
end

MCS = pad_MCS{1,1};
for c = 2:numel(CN)
    nc = size(pad_MCS{1,c},3);
    MCS(:,:,end+1:end+nc) = pad_MCS{1,c};
end



end