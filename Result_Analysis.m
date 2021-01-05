% AUTHOR INFORMATION AND SCRIPT OVERVIEW, V1.0, 12/2020
%{
Author:________________________________________Fabio Vulpi (Github: Ph0bi0) 

                                 PhD student at Polytechnic of Bari (Italy)
                     Researcher at National Research Council of Italy (CNR)

This script performs result analysis of the models trained and tested in
the script MAIN 
%}
%-------------------------------------------------------------------------%

clear all, close all, clc
% Determine where your m-file's folder is.
folder = fileparts(which(mfilename)); 
% Add that folder plus all subfolders to the path.
addpath(genpath(folder));

% automatically attaches to your working directory
ResultDir = strcat(folder,'/results/');

% CHOOSE RESULT
%{
The user can choose the results to analyse setting "ResultName" among the
files stored in the directory "./results".
Results will be automatically loaded in the workspace as a struct "RES"
previously generated running the MAIN script. 
%}
ResultName = 'TrainingResults_1';
load(strcat(ResultDir,ResultName,'.mat'))

% return
%% PERFORMANCE METRICS LINE PLOTS
%{
The following function returns 4 figures containing the analysis of the
four performance metrics:
    - Accuracy 
    - Sensitivity
    - Precision
    - F1 score
The script loads the results of the training generated in the MAIN script
and automatically passes the workspace saved struct "RES" to the function
"PerformanceMetrics_LinePlot".
The function "PerformanceMetrics_LinePlot" closes all open figures and
generates the 4 figures for performance metrics across all the tested
sampling windows. 
The function "PerformanceMetrics_LinePlot" is stored in the directory
"./functions/results analysis functions" and has a function overvirew that
the user can read to understand the plot process.

The user can easily modify ceratain aspects of the plots by changing
numbers in the following fields of the struct "PMLprop":
    - MarkerSize, the size of the markers used for distinguish models
    - LineWidth, the width of lines in the plot
    - MinPerc, the minimum percentage to show on every plot
    - PercRes1, the principal percentage resolution to highlight with a
    solid black line 
    - PercRes2, the secondary percentage resolution to highlight with a
    dashed grey line 
%}

PMLprop.MarkerSize = 12;
PMLprop.LineWidth = 2;
PMLprop.FontSize = 24;
PMLprop.MinPerc = 40;
PMLprop.PercRes1 = 10;
PMLprop.PercRes2 = 5;

PerformanceMetrics_LinePlot(RES,PMLprop)

%% CONFUSION MATRIX PLOT

%{

%}
CMprop.FontSize = 15;
CMprop.Normalization = 'row-normalized';

ConfusionMatrix_Plot(RES,CMprop)







