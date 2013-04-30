%Same non-linearity g used throughout


M = 5;   %Number of units per layer
N = 3; %Number of layers

%Call non-linearity
g = @nonLin;

%Assuming only one signal 
s = randn(M,1);

%Init one possible correct set of weights
Wsoln = randn(M,M,N-1);

%Compute an output value the function can attain (at least with WCorr)
ySoln = propSig(1,N,Wsoln,s)


%Now we initialize the network
W = randn(M,M,N-1);

%Initialize network of Neurons
x = zeros(M,N);
x(:,1) = s;
for i=2:N
    x(:,i)=propSig(1,i,W,s);
end


numBP=1000; %Number of times to repeat backpropagation
bpStep=.001; %backprop step size

%Not currently working (clearly wrong for linear) - Find bug.
for cnt=1:numBP
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
    norm(ySoln-out)
end







