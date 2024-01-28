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

sensors = fieldnames(Channels);

datadir = dir(DataDir);
for i = 1:numel(datadir)
    terrainName = datadir(i).name;
    switch terrainName
        case {'.','..','.DS_Store'}
        otherwise
            TerDir = dir(strcat(DataDir,'/',terrainName));
            for j = 1:numel(TerDir)
                matFName = TerDir(j).name;
                switch matFName
                    % For every file in terrain
                    case {'.','..','.DS_Store'}
                    otherwise
                        us = find(matFName=='_'); % undescore
                        dt = find(matFName=='.'); % dot
                        for k = 1:numel(sensors)
                            switch matFName(1:us-1)
                                case sensors{k}
                                    d = load(strcat(DataDir,'/',terrainName,'/',matFName));
                                    f = fieldnames(d);
                                    file_idx = str2double(matFName(us+1:dt-1));
                                    Data.(terrainName){file_idx,k} = d.(f{1});
                            end
                        end
                end
            end
    end
end

terrain_names = fieldnames(Data);


for i = 1:numel(terrain_names)
    terrain = terrain_names{i};
    for j = 1:size(Data.(terrain),1)
        for c = 1:numel(sensors)
            sensor = sensors{c};
            on = find(cell2mat(Channels.(sensor).on(:,2))==1);
            REC.time.(terrain){j,c} = Data.(terrain){j,c}(:,1);
            REC.data.(terrain){j,c} = Data.(terrain){j,c}(:,on+1);
        end
    end
end


end