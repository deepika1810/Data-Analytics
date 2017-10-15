load spambase.data
features = spambase(:,1:57); % loading the features
cls = spambase(:,58); %loading the classes of the observations

[B,FitInfo]=lassoglm(features,cls,'normal','CV',10);

lassoPlot(B,FitInfo,'plottype','CV');
