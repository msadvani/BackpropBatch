%Try to do this computation the simplest way to begin, by performing 3
%seperate averages over some time interval T

%Same non-linearity g used throughout
M = 5;   %Number of units per layer
N = 3; %Number of layers
T=50; %Number of timesteps to perform averaging over before recomputing weights
epsilon=.1; %standard dev of noise

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
for t =2:T
    x(:,:,t)=pNS1T(x(:,:,t-1),s,W,noise(:,:,t));
end

x(:,:,T)


xCheck = propNoisySigT(x(:,:,1),s,W,noise,T);

xCheck(:,:,T)



%Now average the appropriate quantities

deltaX = mean(repmat(ySoln,[1,1,T])- x(:,N,:),3);


% temp = x(:,:,T);
% x = zeros(M,N,T);
% x(:,:,1) = temp
% for t =2:T
%     x(:,:,t)=propNoisySig1T(x(:,:,t-1),s,W,noise(:,:,t));
% end



