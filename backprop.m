function [err] = backprop(s,NumLayers, bpStep, numBP)
%Input one (in future generalize to more inputs)
%E.g. initialize as backprop(randn(5,1),3,.01,10000)

err= zeros(1,numBP);
M = size(s,1);
N= NumLayers;

g = @nonLin;

%Init one possible correct set of weights
Wsoln = randn(M,M,N-1);

%Compute an output value the function can attain (at least with WCorr)
ySoln = propSig(1,N,Wsoln,s);


%Now we initialize the network
W = randn(M,M,N-1);

%Initialize network of Neurons
x = zeros(M,N);
x(:,1) = s;
for i=2:N
    x(:,i)=propSig(1,i,W,s);
end


%numIter=10000; %Number of times to repeat backpropagation
%bpStep=.01; %backprop step size

for cnt=1:numBP
    [cnt,numBP];
    x(:,1) = s;
    for i=2:N
        x(:,i)=propSig(1,i,W,s);
    end 
    
    
    dW = zeros(M,M,N-1);
    
    %Deltas computed for each layer
    delta = zeros(M,N-1);
    
    [y,yp] = g(x(:,N));
    delta(:,N-1) = yp.*(ySoln-x(:,N));
   
    for k=N-1:2
        [y,yp] = g(x(:,k)); 
        delta(:,k-1) = yp.*(W(:,:,k)'*delta(:,k));
    end
   
    
    for m=1:N-1
        dW(:,:,m)=delta(:,m)*x(:,m)';
    end
    
    W = W+bpStep*dW;
    
    out = propSig(1,N,W,s);
    err(cnt)=norm(ySoln-out);
    
    norm(ySoln-out)
end

end

