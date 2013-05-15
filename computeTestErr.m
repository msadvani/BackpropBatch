%Run tests on the different backpropagation algorithms over different
%classes of networks, etc. for a detailed comparison.

clear all;
close all;

 
dataDim=5;
M=dataDim;

numTrainEx =30;
numTestEx = 300;


numLayers=3;
N= numLayers;
rng(5005)
%rng(5) %works well here, why?

%Init one possible correct set of weights
Wsoln = (1/sqrt(M))*randn(M,M,N-1);


input = randn(dataDim,numTrainEx);
 
 %To optimize, want to find the step size to converge the fastest (so need
 %to modify functions: numIter-> maxIter and find a tolerance at which you can stop)
 
 numIter = 500;
 stepSz=.01;
 
 seed=9;
 Tavg = 15;
 
 
[err,errSet, Wbp,WbpTime]=backprop(input,numLayers,stepSz,numIter,Wsoln, seed);
 
[err1, WlocBP, WlocBPTime] = localNoisyBPSim(input,numLayers,sqrt(stepSz),1,Tavg, numIter, Wsoln, seed);




%now perform cross validation based on more examples


 
testInput = randn(dataDim,numTestEx);
 
ySolnSet = propSig(1,N,Wsoln,testInput);

errBP = zeros(1,numIter);
errLocBP = zeros(1,numIter);
for cnt=1:numIter
yBP = propSig(1,N,WbpTime(:,:,:,cnt),testInput);
errBP(cnt)=norm(yBP-ySolnSet,'fro')^2;

ylocBP = propSig(1,N,WlocBPTime(:,:,:,cnt),testInput);
errLocBP(cnt) = norm(ylocBP-ySolnSet,'fro')^2;


end

plot([1:numIter],errBP);
hold on;
plot([1:numIter],errLocBP,'r--');

legend('bp','local bp');
title('test error');


%% Static training error
% numTestEx = 10;
%  
% testInput = randn(dataDim,numTestEx);
%  
% ySolnSet = propSig(1,N,Wsoln,testInput);

% yBP = propSig(1,N,Wbp,testInput);
% 
% disp('belief prop error')
% norm(yBP-ySolnSet)^2
% 
% ylocBP = propSig(1,N,WlocBP,testInput);
%  
%  disp('local bp error')
% norm(ylocBP-ySolnSet)^2


 
 
 %% TO DO: Set up bp's to output weights and run a non-noisy error test. This is just a check that everything is working

 %After outputting W's run a non-noisy test of generalization error? (check
 %if this is what Ben did)
 
 %Also can set eta to automatically update to optimize learning rate
 %(particularly if working on a bigger problem...)
 
