%Run tests on the different backpropagation algorithms over different
%classes of networks, etc. for a detailed comparison.

clear all;
close all;

 
dataDim=5;
M=dataDim;

numTrainEx =25;

numLayers=3;
N= numLayers;
rng(6)

%Init one possible correct set of weights
Wsoln = (1/sqrt(M))*randn(M,M,N-1);


input = randn(dataDim,numTrainEx);
 
 %To optimize, want to find the step size to converge the fastest (so need
 %to modify functions: numIter-> maxIter and find a tolerance at which you can stop)
 
 numIter = 500;
 stepSz=.01;
 
 seed=7;
 Tavg = 50;
 
 
[err,errSet, Wbp]=backprop(input,numLayers,stepSz,numIter,Wsoln, seed);
 
[err1, WlocBP] = localNoisyBPSim(input,numLayers,sqrt(stepSz),1,Tavg, numIter, Wsoln, seed);

plot([1:numIter],err);
hold on;
plot([1:numIter],err1,'r--')



%now perform cross validation based on more examples

% numTestEx = 10;
% 
% testInput = randn(dataDim,numTestEx);
% 
% yTestSolnSet = propSig(1,N,Wsoln,testInput);
% 







%propSig(1,N,Wout,input);




%plot([1:numIter],energy1,'b--')



%title('Backprop error')




%legend('exact bp','noisy bp error','noisy bp energy')

 
 
 
 %% TO DO: Set up bp's to output weights and run a non-noisy error test. This is just a check that everything is working

 %After outputting W's run a non-noisy test of generalization error? (check
 %if this is what Ben did)
 
 %Also can set eta to automatically update to optimize learning rate
 %(particularly if working on a bigger problem...)
 
