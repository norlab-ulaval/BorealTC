function [mSpect,pSpect,TimeGrid,FreqGrid] = Spectrogram(Signal,TimeStamps,sf,tw,to)

% FUNCTION OVERVIEW
%{
This function computes the spectrogram of the signal contained in the array
"Signal" according to the time window "tw" and the time overlap "to".
Returns:
    - the magnitude spectrogram in the matrix "mSpect"
    - the corresponding phase spectrogram in the matrix "pSpect"
    - the time grid (useful for plots) in the matrix "TimeGrid"
    - the frequency grid (useful for plots) in the matrix "FreqGrid"
Uses the information of the sampling frequency at which the signal is
provided contained as a double in the input variable "sf".
%}

t0 = TimeStamps(1);
t1 = t0+tw;
k = 1;
while t1 <= TimeStamps(end)
    TimePoints(k,1) = t1;
    [~,e0] = min(abs(t0-TimeStamps));
    [~,e1] = min(abs(t1-TimeStamps));
    [mSpect(:,k),pSpect(:,k),Freq] = DFT(Signal(e0:e1),sf);
    k = k + 1;
    t0 = t1-to;
    t1 = t0+tw;
end

[TimeGrid,FreqGrid]= meshgrid(TimePoints,Freq);

end