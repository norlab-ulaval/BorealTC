function REC = Get_Recordings(DataDir,Channels)

% FUNCTION OVERVIEW
%{
this function extracts the data from DataDir folder and returns a struct
"REC" with 2 fields:
    - data, containing the numerical data of the recordings
    - time, containing the timestamps of the recordings
Each one of the 2 fields of "REC" is a struct with has as many fields as
available terrains (e.g. REC.data.<terrain_label>)
Every field <terrain_label> contains a cell with as many rows as available
recordings and as many colums as different sampling frequencies (here imu
50Hz and pro data 15Hz)
%}

CN = fieldnames(Channels);

datadir = dir(DataDir);
for i = 1:numel(datadir)
    switch datadir(i).name
        case {'.','..','.DS_Store'}
        otherwise
            TerDir = dir(strcat(DataDir,'/',datadir(i).name));
            for j = 1:numel(TerDir)
                switch TerDir(j).name
                    case {'.','..','.DS_Store'}
                    otherwise
                        us = find(TerDir(j).name=='_');
                        dt = find(TerDir(j).name=='.');
                        for k = 1:numel(CN)
                            switch TerDir(j).name(1:us-1)
                                case CN{k}
                                    d = load(strcat(DataDir,'/',datadir(i).name,'/',TerDir(j).name));
                                    f = fieldnames(d);
                                    Data.(datadir(i).name){str2double(TerDir(j).name(us+1:dt-1)),k} = d.(f{1});
                            end
                        end
                end
            end
    end
end

FN = fieldnames(Data);


for i = 1:numel(FN)
    for j = 1:size(Data.(FN{i}),1)
        for c = 1:numel(CN)
            on = find(cell2mat(Channels.(CN{c}).on(:,2))==1);
            REC.time.(FN{i}){j,c} = Data.(FN{i}){j,c}(:,1);
            REC.data.(FN{i}){j,c} = Data.(FN{i}){j,c}(:,on+1);
        end
    end
end


end