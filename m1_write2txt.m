% Prepare data for log-rank test and Kaplan-Meier survival curve

clear

tab = readtable('SurvivalData-AAC-20210901.xlsx');
% tab.group = 1 represents treatments designed for PAC.

dlmwrite('res_m1.txt', [tab.os, tab.status, tab.group], '\t');