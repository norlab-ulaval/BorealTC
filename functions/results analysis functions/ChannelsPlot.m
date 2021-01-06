function ChannelsPlot(RES)

% FUNCTION OVERVIEW
%{
This function displays the available channels of the results together with
their labels.
The user can keep track of the active channels to which the following
results correspond.
%}

FS = 20;
cmap = [0.6350,0.0780,0.1840;0.4660,0.6740,0.1880];

CN = fieldnames(RES.Channels);
figure('Name','SENSOR CHANNELS VISUALIZATION')

for c = 1:numel(CN)
    subplot(numel(CN)+1,1,c)
    hold on
    Mat = (cell2mat(RES.Channels.(CN{c}).on(:,2)))';
    rows = 1:size(Mat,2);
    cols = (1:size(Mat,1))';
    Colors = zeros(size(Mat)+[1 1]);
    Colors(1:end-1,1:end-1)=Mat;
    if length(rows) > 1
        rows = [2*rows(1)-rows(2),rows];
    else
        rows = [0,rows];
    end
    if length(cols) > 1
        cols = [2*cols(1)-cols(2);cols];
    else
        cols = [0,cols];
    end
    [R,C] = meshgrid(rows,cols);
    pcolor(R,C,Colors)
    colormap(cmap)
    xlim([rows(1) rows(end)])
    ylim([cols(1) cols(end)])
    xticks([])
    yticks([])
    title({strcat('Group identification :'," ",CN{c}),...
        strcat('Sampling frequency :'," ",num2str(RES.Channels.(CN{c}).sf)," ",'[Hz]')},...
        'FontSize',FS+5)
    xtxt = zeros(1,max(rows));
    for i = 2:numel(rows)
        xtxt(i-1) = (rows(i-1)+rows(i))/2;
    end
    ytxt = 0.5*ones(size(xtxt));
    text(xtxt,ytxt,RES.Channels.(CN{c}).on(:,1),'HorizontalAlignment','center','FontSize',FS)
    clear rows cols Mat Colors R C
end

subplot(numel(CN)+1,1,numel(CN)+1)
hold on
Mat = [1,0];
rows = 1:size(Mat,2);
rows = rows*0.10;
cols = (1:size(Mat,1))';
cols = cols*0.10;
Colors = zeros(size(Mat)+[1 1]);
Colors(1:end-1,1:end-1)=Mat;
if length(rows) > 1
    rows = [2*rows(1)-rows(2),rows];
else
    rows = [0,rows];
end
if length(cols) > 1
    cols = [2*cols(1)-cols(2);cols];
else
    cols = [0,cols];
end
[R,C] = meshgrid(rows,cols);
pcolor(R,C,Colors)
colormap(cmap)
xlim([rows(1) rows(end)])
ylim([cols(1) cols(end)])
xticks([])
yticks([])
xtxt = zeros(1,numel(rows)-1);
for i = 2:numel(rows)
    xtxt(i-1) = (rows(i-1)+rows(i))/2;
end
ytxt = 0.5*max(cols)*ones(size(xtxt));
text(xtxt,ytxt,{'ON','OFF'},'HorizontalAlignment','center','FontSize',FS)
axis equal
set(gca,'color','none')
axis off
title('Legend','FontSize',FS)
clear rows cols Mat Colors R C

sgtitle('Available sensor channels','FontSize',FS+10)

end