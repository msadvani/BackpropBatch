%Run tests on the different backpropagation algorithms over different
%classes of networks, etc. for a detailed comparison.
clear all;
close all;

 
dataDim=5;
M=dataDim;

numEx =1;

numLayers=3;
N= numLayers;
rng(5)
input = randn(dataDim,numEx);
 
 %To optimize, want to find the step size to converge the fastest (so need
 %to modify functions: numIter-> maxIter and find a tolerance at which you can stop)
 
 numIter = 350;
 stepSz=.01;
 
 seed=17;
 Tavg = 500;
 
 
 [err,errSet, W]=backprop(input,numLayers,stepSz,numIter,seed);

 plot([1:numIter],err,'g');
 
 hold on;
 
  
 
err1 = localNoisyBPSim(input,numLayers,sqrt(stepSz),1,Tavg, numIter,seed);
plot([1:numIter],err1,'r--')


err2 = localNoisyBPSimOld(input,numLayers,sqrt(stepSz),1,Tavg, numIter,seed);
 
plot([1:numIter], err2,'b--')



title('W11, T=20000')
legend('exact bp','new noisy bp','old noisy bp')

 
 
 
 %% TO DO: Set up bp's to output weights and run a non-noisy error test. This is just a check that everything is working

 %After outputting W's run a non-noisy test of generalization error? (check
 %if this is what Ben did)
 
 %Also can set eta to automatically update to optimize learning rate
 %(particularly if working on a bigger problem...)
 
