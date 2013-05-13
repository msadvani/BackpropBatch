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


% baseStep = .01;
% [errNonLoc]=backprop(randn(5,1),10,baseStep,3000);
% %plot([1:length(errNonLoc)],errNonLoc)
% 
% hold on;
% %[errLoc]=localNoisyBPSim(randn(5,1),10,sqrt(baseStep),1,10,3000);
% [errLoc]=localNoisyBPSimAnneal(randn(5,1),10,sqrt(baseStep),.4,1,40,3000);
% %For a fair comparison step size = epislon^2 *gradstep_noisy
%  plot([1:length(errNonLoc)],errNonLoc,'bo',[1:length(errLoc)],errLoc,'r+')
% 
% xlabel('Number of Gradient Steps')
% ylabel('Error')
% title('Comparison of Backprop Algorithms')
% 
% legend('normal backprop','local noisy backprop');


%% Just for testing things work


%backpropOnline(rand(10,5),10,.01,10000); 


%Local version seems more robust to depth of network. Not really sure why,
%Maybe because it doesn't change a layer until it is noticed a correlation,
%so fewer false positives (May work signicantly better in deeper networks


%localNoisyBPSimOnline(rand(10,5),3,.1,1,50,10000);

%localNoisyBPSimOnline(rand(10,5),10,.1,1,50,10000);
%backpropOnline(rand(10,5),40,.01,10000); 


 %numTSteps = 100;
 %err = localNoisyBPSimOnline(rand(10,5),10,.1,1,40, numTSteps);
 %plot([1:numTSteps],err)



%Try writing an online version of noisy backprop, probably the noise helps
%avoid local minima, and may be better than vanilla backprop. I don't know
%how this might change if I added a momentum term...



%% Try optimizing backprop and local backprop given the same input data


% dataDim=8;
% numEx = 1;
% 
% numLayers=11;
% %input = rand(dataDim,numEx);

% %input = randn(dataDim,numEx);
% input = randn(dataDim,numEx);
% 
% %To optimize, want to find the step size to converge the fastest (so need
% %to modify functions: numIter-> maxIter and find a tolerance at which you can stop)
% 
% numIter = 200;
% stepSz=.01;
% 
% 
% err=backpropOnline(input,numLayers,stepSz,numIter);
% plot([1:numIter],err);
% 
% hold on;
% 
% [err,errSet] = localNoisyBPSimOnline(input,numLayers,sqrt(stepSz),1,400, numIter);
% 
%  plot([1:numIter],err,'r--')
 
 

%% Repeat, but initialize with a solution
 %%%%why isn't it initializing at the same point??
 
dataDim=10;
M=dataDim;

numEx = 10;

numLayers=6;
N= numLayers;

input = randn(dataDim,numEx);
 
 %To optimize, want to find the step size to converge the fastest (so need
 %to modify functions: numIter-> maxIter and find a tolerance at which you can stop)
 
 numIter = 350;
 stepSz=.01;
  

 seed=6;
 
 
 err=backprop(input,numLayers,stepSz,numIter,seed);
 plot([1:numIter],err);
 
 hold on;
 
%[err,errSet] = localNoisyBPSim(input,numLayers,.5*sqrt(stepSz),4,150*4, numIter,seed);

%In the high noise low step case, it will converge quite quickly (fast averaging -- low T), but the system is so
%noisy, on average it will have poor performance...

%[err,errSet] = localNoisyBPSim(input,numLayers,1,stepSz,1000, numIter,seed);

%[err,errSet] = localErfNoisyBPSim(input,numLayers,sqrt(stepSz),1,100, numIter,seed);

%plot([1:numIter],err,'r--')

[err,errSet] = localErfNoisyBPSim(input,numLayers,sqrt(stepSz),1,100, numIter,seed);

plot([1:numIter],err,'g--')

[err,errSet] = localNoisyBPSimOldCopy(input,numLayers,1,sqrt(stepSz),100, numIter,seed);

plot([1:numIter],err,'r--')


%[err,errSet] = localNoisyBPSimOnlineInit(input,numLayers,sqrt(stepSz),1,1000, numIter,Wsoln, Winit);
 
%[err] = localNoisyBPSepInit(input,numLayers,sqrt(.01),1,1000, numIter,Wsoln, Winit); 



