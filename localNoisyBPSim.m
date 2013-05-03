function [err] = localNoisyBPSim(s,NumLayers, epsilon, gradStep, Tavg, numIter)
%Input one (in future generalize to more inputs)

err = zeros(1,numIter);
M = size(s,1);
N= NumLayers;
T = Tavg;
%numIter = 5000; (provided by input)- Note make this a maxIter and add a
%convergence threshold

layUp=[2:N]; %Set of layers to update
%Create non-linearity handle
g = @nonLin;

%Assuming only one signal 
%s = randn(M,1);

%Init one possible correct set of weights
Wsoln = randn(M,M,N-1);

%Compute an output value the function can attain (at least with WCorr)
ySoln = propSig(1,N,Wsoln,s);

%Now we initialize the network
W = randn(M,M,N-1);

%Initialize network of Neurons (for the whole time window)
x = zeros(M,N,T);

x(:,1,1) = s;

noise = epsilon*randn(M,N,T);

x = propNoisySig(x(:,:,1),s,W,noise,T);

%Now average the appropriate quantities



for cnt=1:numIter
    [cnt,numIter]
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
    err(cnt) = norm(deltaX); %print average error from target
    

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

%err = norm(deltaX);



end

