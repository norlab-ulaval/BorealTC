function [Magn,Phase,Freq] = DFT(Signal,sf)

% FUNCTION OVERVIEW
%{
This function computes the Single Sided Discrete Fourier Transform of a
singal "Signal" and returns the magnitude of the spectrum in the array
"Magn", the phase in the array "Phase" and the frequency span in the array 
"Freq".
%}

npf = length(Signal); % number of points for frequency
DSFT = fft(Signal); % Double Sided Discrete Fourier Transform
if size(DSFT,2)~=1
    DSFT = DSFT';
end
m = abs(DSFT); % magnitude of complex numbers
DSFT(m<1e-6) = 0; %clean values smaller than 1e-6
Phase = unwrap(angle(DSFT));

% the procedure is different depending if the number of points is even or
% odd

if rem(npf,2)==0
    SSFT = DSFT(1:ceil((npf+1)/2))/npf;
    SSFT(2:end-1) = 2*SSFT(2:end-1);
else
    SSFT = DSFT(1:ceil((npf+1)/2))/npf;
    SSFT(2:end) = 2*SSFT(2:end);
end


Magn = (abs(SSFT));
Phase = (Phase(1:length(SSFT)));

Freq = (linspace(0,sf/2,length(SSFT)))';

        
end