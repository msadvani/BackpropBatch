%Try to do this computation the simplest way to begin, by performing 3
%seperate averages over some time interval T

%Same non-linearity g used throughout
M = 5;   %Number of units per layer
N = 3; %Number of layers
T=300; %Number of timesteps to perform averaging over before recomputing weights
epsilon=.1; %standard dev of noise
gradStep = 1; %gradient stepSize

%Create non-linearity handle
g = @nonLin;

%Assuming only one signal 
s = randn(M,1);

%Init one possible correct set of weights
Wsoln = randn(M,M,N-1);

%Compute an output value the function can attain (at least with WCorr)
ySoln = propSig(1,N,Wsoln,s);

%Now we initialize the network
W = randn(M,M,N-1);

%Initialize Noise
noise = epsilon*randn(M,N,T);

%Initialize network of Neurons (for the whole time window)
x = zeros(M,N,T);

x(:,1,1) = s;

x = propNoisySig(x(:,:,1),s,W,noise,T);

%Now average the appropriate quantities

numIter = 1000;

for cnt=1:numIter
    temp = x(:,:,N);
    x = propNoisySig(temp,s,W,noise,T);

    deltaX = mean(repmat(ySoln,[1,1,T])- x(:,N,:),3);
    norm(deltaX)

    c=2; %updating w_(c-1)

    tSet = 1:T-N+c;
    correl = zeros(M,M);
    for t=tSet
        correl = correl+ x(:,N,t+(N-c))*noise(:,c,t)';
    end
    correl = correl/length(tSet); 

    dW_R = correl'*deltaX;

    temp = x(:,:,N);
    x = propNoisySig(temp,s,W,noise,T);

    dW = gradStep*dW_R*mean(x(:,c-1,:),3)';

    W(:,:,c-1)= W(:,:,c-1)+dW;
    %x = propNoisySig(x(:,:,N),s,W,noise,N); %push effects of new weights to the end of the network
    x = propNoisySig(x(:,:,N),s,W,noise,T); %push effects of new weights to the end of the network
    
end




% temp = x(:,:,T);
% x = zeros(M,N,T);
% x(:,:,1) = temp
% for t =2:T
%     x(:,:,t)=propNoisySig1T(x(:,:,t-1),s,W,noise(:,:,t));
% end




