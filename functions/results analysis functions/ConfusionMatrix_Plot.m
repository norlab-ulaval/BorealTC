function ConfusionMatrix_Plot(RES,CMprop)

% FUNCTION OVERVIEW
%{
This function plots confusion matrixes relative to the results contained in
the input struct "RES".
This function returns 2 figures:
    - the first shows overall confusion matrixes of k-fold cross validation
    process of every sampling window and every model available in "RES"
    - the second shows the confusion matrixes of every model corresponding
    to a choosen sampling window specified in the field "FixedSampWind" of
    the input struct "CMprop".
Aspect of confusion matrixes and normalization properties are regulated by
the input struct "CMprop".
Confusion matrix plots are build using the MATLAB function "confusionchart"
and show true terrain lables on rows and predicted terrain labels on
columns. This is because the input confusion matrixes contained in "RES"
are build in this way and must be built in this way for coherence of the
code.
%}

RN = fieldnames(RES); % Results Names
MN = RN(1:end-2); % Model Names
WN = fieldnames(RES.(MN{1}));

% plot every confusion matrix for every model and everly sampling window
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
            title({MN{j},strcat('Sampling Window'," ",num2str(str2double(WN{i}(us+1:ms(end)-1))/1000)," ",'[s]')})
        else
            title(strcat('Sampling Window'," ",num2str(str2double(WN{i}(us+1:ms(end)-1))/1000)," ",'[s]'))
        end 
        n = n+1;
    end
end

% find the preferred sampling window among those available or disable the
% second plot 
index = [];
for i = 1:numel(WN)
    us = find(WN{i}=='_'); % undescore
    ms = find(WN{i}=='m'); % ms for microseconds
    w = str2double(WN{i}(us+1:ms(end)-1))/1000;
    switch w
        case CMprop.FixedSampWind
            index = i;
    end
end

if ~isempty(index)
    
    % find the proper number of rows and columns for the subplot division
    nr = sqrt(numel(MN));
    if floor(nr) == nr
        nc = nr;
    else
        nr = ceil(nr);
        nc = 1;
        while nr*nc < numel(RES.TerLabls)
            nc = nc + 1;
        end
    end

    figure('Name',strcat('CONFUSION MATRIX FOR SAMPLING WINDOW ='," ",num2str(CMprop.FixedSampWind)," ",'[s]'))
    n = 1;
    for i = 1:numel(MN)
        subplot(nr,nc,n)
        confusionchart(RES.(MN{i}).(WN{index}).ConfusionMat,RES.TerLabls,...
            'Normalization',CMprop.Normalization,'FontSize',CMprop.FontSize+5,...
            'YLabel','True Terrain Labels','XLabel','Predicted Terrain Labels');
        title({MN{i},strcat('Sampling Window'," ",num2str(str2double(WN{index}(us+1:ms(end)-1))/1000)," ",'[s]')})
        n = n+1;
    end
end




end