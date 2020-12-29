function pad = matrix_padding(mat,Nsz)

% FUNCTION OVERVIEW
%{
This function performs padding on a generic input matrix "mat" to match the
size specified by the array "Nsz".
This function can only pad one size of the matrix at the time, to pad both
sizes run this function two time specifiyng each time a new final size
"Nsz".
This function repeats rows or columns of the input matrix "mat" as many
times as needed to match the size "Nsz".
%}

Osz = size(mat); % Old Size
pad = zeros(Nsz); % Padded matrix inizialized to be the requested New Size

if Osz(1) == Nsz(1)
    if Nsz(2) > Osz(2)
        rep = round(Nsz(2)/Osz(2));
        k = 1;
        for j = 1:Osz(2)
            for i = 1:rep
                if k > Nsz(2)
                    break
                end
                pad(:,k) = mat(:,j);
                k = k+1;
            end
        end
        if k <= Nsz(2)
            while k <= Nsz(2)
                pad(:,k) = mat(:,end);
                k = k+1;
            end
        end
    else
        disp('padded size must be greater than older size')
    end
elseif Osz(2) == Nsz(2)
    if Nsz(1) > Osz(1)
        rep = round(Nsz(1)/Osz(1));
        k = 1;
        for j = 1:Osz(1)
            for i = 1:rep
                if k > Nsz(1)
                    break
                end
                pad(k,:) = mat(j,:);
                k = k+1;
            end
        end
        if k <= Nsz(1)
            while k <= Nsz(1)
                pad(k,:) = mat(end,:);
                k = k+1;
            end
        end
    else
        disp('padded size must be greater than older size')
    end
else
    disp('padding procedure can be done along rows or columns not both')
    disp('pad one dimension at the time')

end