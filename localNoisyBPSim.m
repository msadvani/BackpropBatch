function [err,W,Wtime] = localNoisyBPSim(input,NumLayers, epsilon, gradStep, Tavg, numIter, Wsoln, randSeed)
%Input one (in future generalize to more inputs)

rng(randSeed) %seed random number generator

err = zeros(1,numIter);
energy = zeros(1,numIter);

M = size(input,1);
N= NumLayers;
T = Tavg;

numEx = size(input,2); %number of examples
errSet = zeros(numIter,numEx);
energySet = zeros(numIter,numEx);


%numIter = 5000; (provided by input)- Note make this a maxIter and add a
%convergence threshold

layUp=[2:N]; %Set of layers to update
%Create non-linearity handle
g = @nonLin;

%Assuming only one signal 
%s = randn(M,1);

%Init one possible correct set of weights
%Wsoln = (1/sqrt(M))*randn(M,M,N-1);

%Compute an output value the function can attain (at least with WCorr)
ySolnSet = propSig(1,N,Wsoln,input);



%Now we initialize the network
W = (1/sqrt(M))*randn(M,M,N-1);


%Initialize network of Neurons (for the whole time window)
x = zeros(M,N,T);

s = input(:,1);
x(:,1,1) = s;

noiseInit = epsilon*randn(M,N,T);

x = propNoisySig(x(:,:,1),s,W,noiseInit,T);

%Now average the appropriate quantities

Wtime = zeros(M,M,N-1,numIter);

for cnt=1:numIter
    [cnt,numIter]
    
    Wtime(:,:,:,cnt) = W;
    out = propSig(1,N,W,input);
    
     dY = ySolnSet - out;
     err(cnt)= norm(dY,'fro')^2;
    
    exSet = randperm(numEx);
    
    dWbatch = zeros(M,M,N-1);
   
    for exCnt = exSet
        
        %Initialize parameters used to compute updates
        dW = zeros(size(W));
        dW_R = zeros(M,N-1);
        correl = zeros(M,M,N-1);
        
        
        s = input(:,exCnt);
        %s
        
        ySoln = ySolnSet(:,exCnt);
        %ySoln
        
        %propagate signal enough to remove old trace information
        for i=1:N
            x(:,:,T)=pNS1T(x(:,:,T),s,W,epsilon*randn(M,N));
        end
         
        %Run algorithm for T timesteps (and store all T)
        noise = epsilon*randn(M,N,T);
        x = propNoisySig(x(:,:,T),s,W,noise,T);

        deltaX = mean(repmat(ySoln,[1,1,T])- x(:,N,:),3);
        
        Energy = .5*sum((repmat(ySoln,[1,1,T])- x(:,N,:)).^2);
        
        
        %Compute updates for each layer      
        for c=layUp;
            
            Eset = Energy(:,:,N-c+1:T); %really a 1 by T matrix
            xPrevSet = x(:,c-1,N-c+1:T);
            
            
            xPrevSet = reshape(xPrevSet,1,M,T-N+c);
            
            corrTerm = repmat(Eset,M,1).*noise(:,c,1:T-(N-c));
            dW(:,:,c-1) = -gradStep*mean(repmat(corrTerm,1,M).*repmat(xPrevSet,M,1),3);
        end
        
        
        dWbatch = dWbatch+dW;
        
       
            
        %These are two different ways of looking at the error, there will
        %always be some average error with noise, but the term we are
        %looking at is the performance of an averaging estimator
        
        %errSet(cnt,exCnt) = mean(Energy);
        
        %energySet(cnt, exCnt) = mean(Energy);
        %errSet(cnt, exCnt) = norm(deltaX)^2;
    end
    
    W = W + dWbatch;
    %Wout(:,:,:,cnt)=W;
        
     
    
        
end


%err =sum(errSet,2);
%energy =sum(energySet,2);



end

