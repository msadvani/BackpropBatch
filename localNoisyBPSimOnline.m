function [err, errSet] = localNoisyBPSimOnline(input,NumLayers, epsilon, gradStep, Tavg, numIter)
%Input one (in future generalize to more inputs)


rng(2) %seed random number generator

err = zeros(1,numIter);

M = size(input,1);
N= NumLayers;
T = Tavg;

numEx = size(input,2); %number of examples
errSet = zeros(numIter,numEx);

%numIter = 5000; (provided by input)- Note make this a maxIter and add a
%convergence threshold

layUp=[2:N]; %Set of layers to update
%Create non-linearity handle
g = @nonLin;

%Assuming only one signal 
%s = randn(M,1);

%Init one possible correct set of weights
Wsoln = (1/sqrt(M))*randn(M,M,N-1);

%Compute an output value the function can attain (at least with WCorr)
ySolnSet = propSig(1,N,Wsoln,input);

%Now we initialize the network
W = (1/sqrt(M))*randn(M,M,N-1);

%Initialize network of Neurons (for the whole time window)
x = zeros(M,N,T);

s = input(:,1);
x(:,1,1) = s;

noise = epsilon*randn(M,N,T);

x = propNoisySig(x(:,:,1),s,W,noise,T);

%Now average the appropriate quantities


for cnt=1:numIter
    [cnt,numIter];
    
    exSet = randperm(numEx);
   
    for exCnt = exSet
        
        s = input(:,exCnt);
        ySoln = ySolnSet(:,exCnt);

        %Initialize parameters used to compute updates
        dW = zeros(size(W));
        dW_R = zeros(M,N-1);

        %propagate signal enough to remove old trace information
        for i=1:N
            x(:,:,T)=pNS1T(x(:,:,T),s,W,epsilon*randn(M,N));
        end

        %Run algorithm for T timesteps (and store all T)
        noise = epsilon*randn(M,N,T);
        x = propNoisySig(x(:,:,T),s,W,noise,T);

        deltaX = mean(repmat(ySoln,[1,1,T])- x(:,N,:),3);
        %need to add an s variable...
        %err(cnt) = norm(deltaX); %print average error from target


        correl = zeros(M,M,N-1);
      
        for c=layUp; %updating w_(c-1)

            tSet = 1:T-N+c;        
            for t=tSet
                correl(:,:,c-1) = correl(:,:,c-1)+ x(:,N,t+(N-c))*noise(:,c,t)';
            end
            correl(:,:,c-1) = correl(:,:,c-1)/length(tSet); 

            dW_R(:,c-1) = correl(:,:,c-1)'*deltaX;
        end

        for c=layUp
            dW(:,:,c-1) = gradStep*dW_R(:,c-1)*mean(x(:,c-1,:),3)';
        end

        W= W+dW;   
        
    end
    
    %propagate each signal and average over only the set of times where ths
    %input has reached the output
    
    %Number of counts you want to average to show mean prediction of
    %network 
    
    max(max(max(abs(W))))
    
    TavgErr = Tavg;
    
    
    exErr = zeros(1,numEx);
    
    for exNum=1:numEx
        
        xOut= propNoisySig([input(:,exNum),zeros(M,N-1)],input(:,exNum),W,epsilon*randn(M,N,TavgErr),TavgErr);
        avgxOut = mean(xOut(:,N,[N:TavgErr]),3);
        exErr(exNum) =norm(avgxOut-ySolnSet(:,numEx))^2;
    end
    exErr;
    errSet(cnt,:) = exErr;
        
end


err =sum(errSet,2);

end

