function [Accuracy,Sensitivity,Precision,F1score] = ConfMat_PerformanceMetrics(CM)

% FUNCTION OVERVIEW
%{
This function computes Accuracy, Sensitivity, Precision and F1score of
input confusion matrix "CM".
All the above mentioned indexes are returned in percentage.
The confusion matrix given as input must have the true class on rows and
the predicted class on columns.
ACCURACY: is the percentage of samples that have been correctly classified.
SENSITIVITY: for every class is the percentage of true samples for
that class that have been correctly classified.
PRECISION: for every class is the percentage of predicted samples for
that class that have been correctly classified.
F1-SCORE mediates the performance among sensitivity and accuracy.
%}

Accuracy = 100*trace(CM)/sum(sum(CM));

Sensitivity = zeros(size(CM,1),1);
for i = 1:numel(Sensitivity)
    Sensitivity(i) = CM(i,i)/sum(CM(i,:));
end

Precision = zeros(size(CM,1),1);
for i = 1:numel(Precision)
    Precision(i) = CM(i,i)/sum(CM(:,i));
end

F1score = 2*(Precision.*Sensitivity)./(Precision+Sensitivity);

Sensitivity = Sensitivity*100;
Precision = Precision*100;
F1score = F1score*100;

end