%Run tests on the different backpropagation algorithms over different
%classes of networks, etc. for a detailed comparison.
clear all;
close all;

 
dataDim=5;
M=dataDim;

numEx =75;

numLayers=3;
N= numLayers;

input = randn(dataDim,numEx);
 
 %To optimize, want to find the step size to converge the fastest (so need
 %to modify functions: numIter-> maxIter and find a tolerance at which you can stop)
 
 numIter = 350;
 stepSz=.01;
 
 seed=40;
 Tavg = 200;
 
 
 err=backprop(input,numLayers,stepSz,numIter,seed);
 plot([1:numIter],err);
 
 hold on;
 
 
 
err1 = localNoisyBPSim(input,numLayers,sqrt(stepSz),1,Tavg, numIter,seed);

plot([1:numIter],err1,'g--')


%[err,errSet] = localErfNoisyBPSim(input,numLayers,sqrt(stepSz),1,Tavg, numIter,seed);
%plot([1:numIter],err,'g--')

err2 = localErfNoisyBPSim(input,numLayers,sqrt(stepSz),1,Tavg, numIter,seed);
 
 plot([1:numIter],err2,'r--')
 
 legend('bp','old local bp','new local bp')
 
 
 
 
 %% TO DO: Set up bp's to output weights and run a non-noisy error test. This is just a check that everything is working
 %Also need to find out why new local method is doing better and if it is a
 %fluke. It may be related to variance? Or it could be just be due to
 %convexty of error???

 
 %Maybe I should also fiddle with eta to see which one is REALLY faster at
 %optimizing...
 
