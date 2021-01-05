function ConfusionMatrix_Plot(RES,CMprop)

% close all
% FUNCTION OVERVIEW
%{

%}

RN = fieldnames(RES); % Results Names
MN = RN(1:end-2); % Model Names
WN = fieldnames(RES.(MN{1}));

n = 1;
figure('Name','CONFUSION MATRIXES')
for i = 1:numel(WN)
    for j = 1:numel(MN)
        us = find(WN{i}=='_'); % undescore
        ms = find(WN{i}=='m'); % ms for microseconds
        subplot(numel(WN),numel(MN),n)
        confusionchart(RES.(MN{j}).(WN{i}).ConfusionMat,RES.TerLabls,...
            'Normalization',CMprop.Normalization,'FontSize',CMprop.FontSize,...
            'YLabel','True Terrain Labels','XLabel','Predicted Terrain Labels');
        if i == 1
            title({MN{j},strcat('Sampling Window'," ",num2str(str2double(WN{j}(us+1:ms(end)-1))/1000)," ",'[s]')})
        else
            title(strcat('Sampling Window'," ",num2str(str2double(WN{j}(us+1:ms(end)-1))/1000)," ",'[s]'))
        end 
        n = n+1;
    end
end







end