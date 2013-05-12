function [err] = backprop(input,NumLayers, bpStep, numBP,randSeed)
%Input one (in future generalize to more inputs)
%E.g. initialize as backprop(randn(5,1),3,.01,10000)

rng(randSeed); %seed random number generator

err= zeros(numBP,1);
M = size(input,1);
N= NumLayers;

numEx=size(input,2);   %Number of examples

errSet= zeros(numBP,numEx);
g = @nonLin;


%Init one possible correct set of weights
Wsoln = (1/sqrt(M))*randn(M,M,N-1);
%Wsoln = randn(M,M,N-1);


%Compute an output value the function can attain (at least with WCorr)
ySolnSet = propSig(1,N,Wsoln,input);


%Now we initialize the network
W = (1/sqrt(M))*randn(M,M,N-1); %Added term to keep the nonlinearity from getting out of range at initialization
%W = randn(M,M,N-1); (You do need sqrt(M) scaling term!)

s = input(:,1);
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
    orderEx = randperm(numEx);
    %orderEx
    for examp = orderEx
        %set input and output    
        s = input(:,examp);
        ySoln = ySolnSet(:,examp);

        x(:,1) = s;
        for i=2:N
            x(:,i)=propSig(1,i,W,s);
        end 

        dW = zeros(M,M,N-1);

        %Deltas computed for each layer
        delta = zeros(M,N-1);

        [y,yp] = g(x(:,N));
        delta(:,N-1) = yp.*(ySoln-x(:,N));

        for k=N-1:-1:2
            [y,yp] = g(x(:,k)); 
            delta(:,k-1) = yp.*(W(:,:,k)'*delta(:,k));
        end

        for m=1:N-1
            dW(:,:,m)=delta(:,m)*x(:,m)';
        end
        
%         %for testing purposes
%         orig = dW(:,:,1)
%         check=(ySoln-x(:,N));
%         for m=N-1:-1:1
%            check=W(:,:,m)'*check; 
%         end
%         check*x(:,1)'
        
       
        W = W+bpStep*dW;
    end
    
    %Check error of current solution
    out = propSig(1,N,W,input);
    
    dY = ySolnSet - out;
    %currErr = 0; %Keep track of current error
    for i=1:numEx
        errSet(cnt,i) = norm(dY(:,i))^2;
    end
    errSet(cnt,:);
    
    
end
err=sum(errSet,2);
end

