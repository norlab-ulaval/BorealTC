function PerformanceMetrics_LinePlot(RES,PMLprop)

% FUNCTION OVERVIEW
%{
This function analyzes the performance metrics of each model contained in
the input struct "RES" across all the tested sampling windows.
This function returnes 4 figures, 1 for each of the
following performance metrics:
    - Accuracy
    - Sensitivity
    - Precision
    - F1-Score
This 4 performance metrics are computed using the function
"ConfMat_PerformanceMetrics" that the user can find in the directory
"./functions/results analysis functions" together with a function overview.
The sensitivity, precision and f1-score plots have subplots for each of the
available terrain classes specified by the field "TerLabls" of the input
struct "RES".
The four figures are separated in sections.
General appearance of the plots is regulated by the input struct "PMLprop"
%}

RN = fieldnames(RES); % Results Names
MN = RN(1:end-2); % Model Names

LCS = {'#0072BD','#D95319','#EDB120',...
       '#7E2F8E','#77AC30','#4DBEEE','#A2142F'}; % Line Colors

MKS = {'d','s','^','v','h','o','+'}; % Markers

%% FIGURE 1: ACCURACY PLOT 
figure('Name','ACCURACY')
hold on
% analyze the confusion matrix for every sampling window and every model
% store accuracy results and sampling window length
for i = 1:numel(MN)
    WN = fieldnames(RES.(MN{i})); % Window Names
    AT = zeros(size(WN)); % Accuracy Trend initialization
    WL = zeros(size(WN)); % Window Length initialization
    for j = 1:numel(WN)
        [AT(j),~,~,~] = ConfMat_PerformanceMetrics(RES.(MN{i}).(WN{j}).ConfusionMat);
        us = find(WN{j}=='_'); % undescore
        ms = find(WN{j}=='m'); % ms for microseconds
        WL(j) = str2double(WN{j}(us+1:ms(end)-1))/1000; % window length in seconds
    end
    Lp(i) = plot(WL,AT,'Color',LCS{i},'marker',MKS{i},'MarkerSize',PMLprop.MarkerSize,...
        'MarkerFaceColor',LCS{i},'MarkerEdgeColor',LCS{i},'LineWidth',PMLprop.LineWidth);
end

% legend settings
lgd = legend(Lp,MN,'FontSize',PMLprop.FontSize);
lgd.Orientation = 'horizontal';
lgd.Position = [0 0.9276 1 0.0304];
lgd.FontSize = PMLprop.FontSize;

% build the grid with the percentage resolution take the HandleVisibility
% off to keep the legend clean
YG = 0:PMLprop.PercRes1:100;
YDG = PMLprop.MinPerc+PMLprop.PercRes2:PMLprop.PercRes2:100;
for y = 1:length(YG)
    if any(YDG==YG(y))
        YDG(YDG==YG(y)) = [];
    end
end

yticks(YG)
ylim([PMLprop.MinPerc 100])
xticks(WL)
xlim([min(WL),max(WL)])

ax = gca;
XX = ax.XLim;
for y = 1:length(YG)
    p = plot(XX,[YG(y) YG(y)],'-k','LineWidth',2,'HandleVisibility','off');
    p.Color(4) = 1;
    uistack(p,'bottom')
end

for y = 1:length(YDG)
    p = plot(XX,[YDG(y) YDG(y)],'--k','HandleVisibility','off');
    p.Color(4) = 0.5;
    uistack(p,'bottom')
end

for x = 1:length(WL)
    p = plot([WL(x) WL(x)],[YG(1) YG(end)],'-k','HandleVisibility','off');
    p.Color(4) = 0.5;
    uistack(p,'bottom')
end

xlabel('Sample window of analyzed terrain [s]','FontSize',PMLprop.FontSize)
ylabel('Model accuracy [%]','FontSize',PMLprop.FontSize)
title({'ACCURACY',''},'FontSize',PMLprop.FontSize+10)
ax = gca;
ax.FontSize = PMLprop.FontSize;

clear Lp
%% FIGURE 2: SENSITIVITY PLOTS

% find the proper number of rows and columns for the subplot division
nr = sqrt(numel(RES.TerLabls));
if floor(nr) == nr
    nc = nr;
else
    nr = ceil(nr);
    nc = 1;
    while nr*nc < numel(RES.TerLabls)
        nc = nc + 1;
    end
end

figure('Name','SENSITIVITY')
sgtitle('SENSITIVITY','FontSize',PMLprop.FontSize+10)
hold on
% analyze the confusion matrix for every sampling window and every model
% store sensitivity results
for i = 1:numel(MN)
    ST = zeros(numel(RES.TerLabls),size(WN,1)); % Sensitivity Trend initialization
    for j = 1:numel(WN)
        [~,ST(:,j),~,~] = ConfMat_PerformanceMetrics(RES.(MN{i}).(WN{j}).ConfusionMat);
    end

    for k = 1:numel(RES.TerLabls)
        subplot(nr,nc,k)
        hold on
        Lp(i,k) = plot(WL,ST(k,:),'Color',LCS{i},'marker',MKS{i},'MarkerSize',PMLprop.MarkerSize,...
                'MarkerFaceColor',LCS{i},'MarkerEdgeColor',LCS{i},'LineWidth',PMLprop.LineWidth);

        % build the grid with the percentage resolution take the HandleVisibility
        % off to keep the legend clean
        if i == numel(MN)
            YG = 0:PMLprop.PercRes1:100;
            YDG = PMLprop.MinPerc+PMLprop.PercRes2:PMLprop.PercRes2:100;
            for y = 1:length(YG)
                if any(YDG==YG(y))
                    YDG(YDG==YG(y)) = [];
                end
            end
            
            yticks(YG)
            ylim([PMLprop.MinPerc 100])
            xticks(WL)
            xlim([min(WL),max(WL)])

            ax = gca;
            XX = ax.XLim;
            for y = 1:length(YG)
                p = plot(XX,[YG(y) YG(y)],'-k','LineWidth',2,'HandleVisibility','off');
                p.Color(4) = 1;
                uistack(p,'bottom')
            end

            for y = 1:length(YDG)
                p = plot(XX,[YDG(y) YDG(y)],'--k','HandleVisibility','off');
                p.Color(4) = 0.5;
                uistack(p,'bottom')
            end

            for x = 1:length(WL)
                p = plot([WL(x) WL(x)],[YG(1) YG(end)],'-k','HandleVisibility','off');
                p.Color(4) = 0.5;
                uistack(p,'bottom')
            end

            xlabel('Sample window of analyzed terrain [s]','FontSize',PMLprop.FontSize)
            ylabel('Model sensitivity [%]','FontSize',PMLprop.FontSize)
            title({RES.TerLabls{k},''},'FontSize',PMLprop.FontSize)
            ax = gca;
            ax.FontSize = PMLprop.FontSize;
        end
    end
end

% legend settings
lgd = legend(Lp(:,1),MN,'FontSize',PMLprop.FontSize);
lgd.Orientation = 'horizontal';
lgd.Position = [0 0.9033 1 0.0304];

%% FIGURE 3: PRECISION PLOTS

figure('Name','PRECISION')
sgtitle('PRECISION','FontSize',PMLprop.FontSize+10)
hold on
% analyze the confusion matrix for every sampling window and every model
% store precision results
for i = 1:numel(MN)
    PT = zeros(numel(RES.TerLabls),size(WN,1)); % Precision Trend initialization
    for j = 1:numel(WN)
        [~,~,PT(:,j),~] = ConfMat_PerformanceMetrics(RES.(MN{i}).(WN{j}).ConfusionMat);
    end

    for k = 1:numel(RES.TerLabls)
        subplot(nr,nc,k)
        hold on
        Lp(i,k) = plot(WL,PT(k,:),'Color',LCS{i},'marker',MKS{i},'MarkerSize',PMLprop.MarkerSize,...
                'MarkerFaceColor',LCS{i},'MarkerEdgeColor',LCS{i},'LineWidth',PMLprop.LineWidth);

        % build the grid with the percentage resolution take the HandleVisibility
        % off to keep the legend clean
        if i == numel(MN)
            YG = 0:PMLprop.PercRes1:100;
            YDG = PMLprop.MinPerc+PMLprop.PercRes2:PMLprop.PercRes2:100;
            for y = 1:length(YG)
                if any(YDG==YG(y))
                    YDG(YDG==YG(y)) = [];
                end
            end
            
            yticks(YG)
            ylim([PMLprop.MinPerc 100])
            xticks(WL)
            xlim([min(WL),max(WL)])

            ax = gca;
            XX = ax.XLim;
            for y = 1:length(YG)
                p = plot(XX,[YG(y) YG(y)],'-k','LineWidth',2,'HandleVisibility','off');
                p.Color(4) = 1;
                uistack(p,'bottom')
            end

            for y = 1:length(YDG)
                p = plot(XX,[YDG(y) YDG(y)],'--k','HandleVisibility','off');
                p.Color(4) = 0.5;
                uistack(p,'bottom')
            end

            for x = 1:length(WL)
                p = plot([WL(x) WL(x)],[YG(1) YG(end)],'-k','HandleVisibility','off');
                p.Color(4) = 0.5;
                uistack(p,'bottom')
            end

            xlabel('Sample window of analyzed terrain [s]','FontSize',PMLprop.FontSize)
            ylabel('Model precision [%]','FontSize',PMLprop.FontSize)
            title({RES.TerLabls{k},''},'FontSize',PMLprop.FontSize)
            ax = gca;
            ax.FontSize = PMLprop.FontSize;
        end
    end
end

% legend settings
lgd = legend(Lp(:,1),MN,'FontSize',PMLprop.FontSize);
lgd.Orientation = 'horizontal';
lgd.Position = [0 0.9033 1 0.0304];

%% FIGURE 4: F1 SCORE PLOTS

figure('Name','F1-SCORE')
sgtitle('F1-SCORE','FontSize',PMLprop.FontSize+10)
hold on
% analyze the confusion matrix for every sampling window and every model
% store f1 score results
for i = 1:numel(MN)
    FT = zeros(numel(RES.TerLabls),size(WN,1)); % F1 score Trend initialization
    for j = 1:numel(WN)
        [~,~,~,FT(:,j)] = ConfMat_PerformanceMetrics(RES.(MN{i}).(WN{j}).ConfusionMat);
    end

    for k = 1:numel(RES.TerLabls)
        subplot(nr,nc,k)
        hold on
        Lp(i,k) = plot(WL,FT(k,:),'Color',LCS{i},'marker',MKS{i},'MarkerSize',PMLprop.MarkerSize,...
                'MarkerFaceColor',LCS{i},'MarkerEdgeColor',LCS{i},'LineWidth',PMLprop.LineWidth);

        % build the grid with the percentage resolution take the HandleVisibility
        % off to keep the legend clean
        if i == numel(MN)
            YG = 0:PMLprop.PercRes1:100;
            YDG = PMLprop.MinPerc+PMLprop.PercRes2:PMLprop.PercRes2:100;
            for y = 1:length(YG)
                if any(YDG==YG(y))
                    YDG(YDG==YG(y)) = [];
                end
            end
            
            yticks(YG)
            ylim([PMLprop.MinPerc 100])
            xticks(WL)
            xlim([min(WL),max(WL)])

            ax = gca;
            XX = ax.XLim;
            for y = 1:length(YG)
                p = plot(XX,[YG(y) YG(y)],'-k','LineWidth',2,'HandleVisibility','off');
                p.Color(4) = 1;
                uistack(p,'bottom')
            end

            for y = 1:length(YDG)
                p = plot(XX,[YDG(y) YDG(y)],'--k','HandleVisibility','off');
                p.Color(4) = 0.5;
                uistack(p,'bottom')
            end

            for x = 1:length(WL)
                p = plot([WL(x) WL(x)],[YG(1) YG(end)],'-k','HandleVisibility','off');
                p.Color(4) = 0.5;
                uistack(p,'bottom')
            end

            xlabel('Sample window of analyzed terrain [s]','FontSize',PMLprop.FontSize)
            ylabel('Model F1-Score [%]','FontSize',PMLprop.FontSize)
            title({RES.TerLabls{k},''},'FontSize',PMLprop.FontSize)
            ax = gca;
            ax.FontSize = PMLprop.FontSize;
        end
    end
end

% legend settings
lgd = legend(Lp(:,1),MN,'FontSize',PMLprop.FontSize);
lgd.Orientation = 'horizontal';
lgd.Position = [0 0.9033 1 0.0304];
end