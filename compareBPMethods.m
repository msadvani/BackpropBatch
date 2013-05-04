%Run tests on the different backpropagation algorithms over different
%classes of networks, etc. for a detailed comparison.

close all;
clear all;

%% Run a basic comparison error versus iteration

% disp('running normal backprop');
% [errNonLoc]=backprop(randn(5,1),3,.1,5000);
% %plot([1:length(errNonLoc)],errNonLoc)
% 
% %hold on;
% disp('running the local noisy backprop');
% %Note this is based on an error
% [errLoc]=localNoisyBPSim(randn(5,1),3,sqrt(.1),1,10,5000);
% %For a fair comparison step size = epislon^2 *gradstep_noisy
% plot([1:length(errNonLoc)],errNonLoc,'bo',[1:length(errLoc)],errLoc,'r+')


%% Compare error at the end of some fixed iteration length for different noise levels


baseStep = .01;
[errNonLoc]=backprop(randn(5,1),10,baseStep,3000);
%plot([1:length(errNonLoc)],errNonLoc)

hold on;
%[errLoc]=localNoisyBPSim(randn(5,1),10,sqrt(baseStep),1,10,3000);
[errLoc]=localNoisyBPSimAnneal(randn(5,1),40,sqrt(baseStep),.4,1,10,3000);
%For a fair comparison step size = epislon^2 *gradstep_noisy
 plot([1:length(errNonLoc)],errNonLoc,'bo',[1:length(errLoc)],errLoc,'r+')

xlabel('Number of Gradient Steps')
ylabel('Error')
title('Comparison of Backprop Algorithms')

legend('normal backprop','local noisy backprop');
