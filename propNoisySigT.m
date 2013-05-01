function [xOut] = propNoisySigT(X,s,W,noise,T)
% Computes the propagation of a network over T timesteps

M = size(X,1);
N= size(X,2);

xOut = zeros(M,N,T);

prop1=@pNS1T;

xOut(:,:,1)=X;

for t =2:T
    xOut(:,:,t)=prop1(xOut(:,:,t-1),s,W,noise(:,:,t));
end

end

