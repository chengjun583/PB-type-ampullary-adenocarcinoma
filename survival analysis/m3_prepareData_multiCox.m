% Prepare data for multivarible Cox analysis
clear

tab = readtable('SurvivalData-AAC-20210901.xlsx');

time = tab.os;
death = tab.status;

% sex
sex = zeros(size(tab.sex));
indMale = strcmp(tab.sex, 'm');
sex(indMale) = 1;

% differentiation
diff = zeros(size(tab.differentiation));
ind = strcmp(tab.differentiation, 'm2p');
diff(ind) = 1;

data = [time, death, tab.group-1, sex, diff];
dlmwrite('res_m3_data.txt', data, '\t');

