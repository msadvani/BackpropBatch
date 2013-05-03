%Run tests on the different backpropagation algorithms over different
%classes of networks, etc. for a detailed comparison.

close all;
clear all;

disp('running normal backprop');
[errNonLoc]=backprop(randn(5,1),3,.1,5000);
%plot([1:length(errNonLoc)],errNonLoc)

%hold on;
disp('running the local noisy backprop');
%Note this is based on an error
[errLoc]=localNoisyBPSim(randn(5,1),3,sqrt(.1),1,10,5000);
%For a fair comparison step size = epislon^2 *gradstep_noisy
plot([1:length(errNonLoc)],errNonLoc,'bo',[1:length(errLoc)],errLoc,'r+')




