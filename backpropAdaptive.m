function [err,errSet,W, Wtime] = backpropAdaptive(input,NumLayers, bpStep, numBP,Wsoln,randSeed)
%Input one (in future generalize to more inputs)
%E.g. initialize as backprop(randn(5,1),3,.01,10000)


%Temporarily setting these adaptive values here
a=.5*bpStep; 
b=.1;
K=8; %Number of time steps to be considered consistent





rng(randSeed); %seed random number generator

err= zeros(numBP,1);
M = size(input,1);
N= NumLayers;

numEx=size(input,2);   %Number of examples

errSet= zeros(numBP,numEx);
g = @nonLin;



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

Wtime = zeros(M,M,N-1,numBP);

%Error stored for updating parameters
energy = zeros(1,numBP);
lastUpdated = 0;
deltaEnergy = zeros(1,K);

for cnt=1:numBP    
    [cnt,numBP];
    lastUpdated = lastUpdated+1;
    
    
    Wtime(:,:,:,cnt) = W;
    
    %Check error of current solution
    out = propSig(1,N,W,input);
    
    dY = ySolnSet - out;
    
    energy(cnt) = norm(dY,'fro');
    
    
    %currErr = 0; %Keep track of current error
    for i=1:numEx
        errSet(cnt,i) = norm(dY(:,i))^2;
    end
    errSet(cnt,:);
    
    dWbatch = zeros(M,M,N-1);
    
    
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
       
               
        dWbatch = dWbatch+dW;
        
    end
    %cnt
    %energy(1:cnt)
    
    %updating bpStep
    if (cnt>K)
                
        deltaEnergy= energy(cnt-K+1:cnt) - energy(cnt-K:cnt-1);
        
        
        if ( deltaEnergy<0)
            if( lastUpdated>=K)
            bpStep = bpStep +a;
            
            lastUpdated=0;
            end
        elseif (deltaEnergy(K)>0)
            
            bpStep = bpStep*(1-b);
            lastUpdated=0;
            
        end
    end
    
    
    W = W+bpStep*dWbatch;
    
    
        
end
err=sum(errSet,2);
end

