%%

datadir = dir(DataDir);
%%
FN = fieldnames(AugTrain);
TotLabl = [AugTrain.(FN{1}).labl;AugTest.(FN{1}).labl];
NumTer = max(double(TotLabl));
SizTer = zeros(NumTer,1);
for i = 1:NumTer
    SizTer(i) = numel(find(double(TotLabl)==i));
end

%%

[Train,Test] = Partition_Data(REC,Channels,KFOLD,PART_WINDOW,RNG);
w = SAMP_WINDOWS(3);
[AugTrain,AugTest] = Augment_Data(Train,Test,Channels,w,AUG);
%%
sf = Channels.pro.sf;
Signal = AugTrain.folder_1.data{1,1}(:,6);

[Magn,Phase,Freq] = DFT(Signal,sf);
%%
TimeStamps = AugTrain.folder_1.time{1,1};
tw = 0.4;
to = 0.2;

% [mSpect,pSpect,TimeGrid,FreqGrid] = Spectrogram(Signal,TimeStamps,sf,tw,to);

%%
SignalCell = AugTrain.folder_1.data(1,:);
TimeCell = AugTrain.folder_1.time(1,:);
[MCS,TimeGrids,FreqGrids] = Multichannel_Spectrogram(SignalCell,TimeCell,Channels,tw,to);

%%

[TrainMCS,TestMCS] = MCS_Data(AugTrain,AugTest,Channels,tw,to);

%%
[TrainDS,TestDS]= DownSample_Data(AugTrain,AugTest,Channels);

%%

cc = confusionchart(RES.CLSTM.SampWindow_1500ms.ConfusionMat);
cc.RowSummary = 'row-normalized';
cc.ColumnSummary = 'column-normalized';

acc = 0;
for i = 1:size(RES.CLSTM.SampWindow_1500ms.ConfusionMat,1)
    acc = acc + RES.CLSTM.SampWindow_1500ms.ConfusionMat(i,i);
end

acc = 100*(acc/sum(sum(RES.CLSTM.SampWindow_1500ms.ConfusionMat)));
