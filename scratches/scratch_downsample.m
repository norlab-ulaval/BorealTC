function [TrainDS,TestDS]= DownSample_Data(Train,Test,Channels)

% FUNCTION OVERVIEW
%{
This function downsamples the data sampled at higher frequency according to
the lowest sampling frequency. The sampling frequency information is
contained in the input struct "Channels".
This procedure is necessary to pass the time sequences to a Long Short-Term
Memory Recurrent Neural Network.
The function takes as input the partitioned dataset in the structs "Train"
and "Test" and returns two structs "TrainDS" and "TestDS" with the
downsampled data that still keep the original order of sensor channels
expressed in "Channels" and kept in "Train" and "Test".
%}

% find the channel with the lower sampling frequency
CN = fieldnames(Channels);
sf = Channels.(CN{1}).sf;
lc = 1;
for c = 2:numel(CN)
    if Channels.(CN{c}).sf < sf
        sf = Channels.(CN{c}).sf;
        lc = c;
    end
end

% find the sizes of every channel and save them in the array "sz"
sz = zeros(1,numel(CN));
for c = 1:numel(CN)
    on = find(cell2mat(Channels.(CN{c}).on(:,2))==1);
    sz(c) = numel(on);
end

FN = fieldnames(Train);
NumFold = numel(FN);

% downsample the data mantaining channels order
for i = 1:NumFold
    KFold = FN{i};
    for j = 1:size(Train.(KFold).data,1)
        for c = 1:numel(CN)
            if c == 1
                if c == lc
                    TrainDS.(KFold).data{j,1}(:,1:sz(c)) = Train.(KFold).data{j,c};
                else
                    for k = 1:size(Train.(KFold).data{j,lc},1)
                        tm = Train.(KFold).time{j,lc}(k);
                        [~,el] = min(abs(tm-Train.(KFold).time{j,c}));
                        TrainDS.(KFold).data{j,1}(k,1:sz(c)) = Train.(KFold).data{j,c}(el,:);
                    end
                end
            else
                if c == lc
                    TrainDS.(KFold).data{j,1}(:,sum(sz(1:c-1))+1:sum(sz(1:c-1))+sz(c)) = Train.(KFold).data{j,c};
                else
                    for k = 1:size(Train.(KFold).data{j,lc},1)
                        tm = Train.(KFold).time{j,lc}(k);
                        [~,el] = min(abs(tm-Train.(KFold).time{j,c}));
                        TrainDS.(KFold).data{j,1}(k,sum(sz(1:c-1))+1:sum(sz(1:c-1))+sz(c)) = Train.(KFold).data{j,c}(el,:);
                    end
                end
            end
        end
        TrainDS.(KFold).time{j,1} = Train.(KFold).time{j,lc};
    end
    TrainDS.(KFold).labl = Train.(KFold).labl;

    for j = 1:size(Test.(KFold).data,1)
        for c = 1:numel(CN)
            if c == 1
                if c == lc
                    TestDS.(KFold).data{j,1}(:,1:sz(c)) = Test.(KFold).data{j,c};
                else
                    for k = 1:size(Test.(KFold).data{j,lc},1)
                        tm = Test.(KFold).time{j,lc}(k);
                        [~,el] = min(abs(tm-Test.(KFold).time{j,c}));
                        TestDS.(KFold).data{j,1}(k,1:sz(c)) = Test.(KFold).data{j,c}(el,:);
                    end
                end
            else
                if c == lc
                    TestDS.(KFold).data{j,1}(:,sum(sz(1:c-1))+1:sum(sz(1:c-1))+sz(c)) = Test.(KFold).data{j,c};
                else
                    for k = 1:size(Test.(KFold).data{j,lc},1)
                        tm = Test.(KFold).time{j,lc}(k);
                        [~,el] = min(abs(tm-Test.(KFold).time{j,c}));
                        TestDS.(KFold).data{j,1}(k,sum(sz(1:c-1))+1:sum(sz(1:c-1))+sz(c)) = Test.(KFold).data{j,c}(el,:);
                    end
                end
            end
        end
        TestDS.(KFold).time{j,1} = Test.(KFold).time{j,lc};
    end
    TestDS.(KFold).labl = Test.(KFold).labl;

end

end