function [Y] = pNS1T(X,s,W,noise1)
% Computes the propagation of a network, given the network and the signal and noise at that timestep

Y = zeros(size(X));
N= size(X,2);

if (size(W,1)~=size(W,2))
    disp('dimension Mismatch'); 
end
if (size(W,1)~=length(s))
    disp('dimension Mismatch');
end


g = @nonLin;


%Compute the values of the network after propagating one time step
for lay=N:-1:2
    Y(:,lay)= g(W(:,:,lay-1)*X(:,lay-1)+noise1(:,lay));
end
Y(:,1) = s+noise1(:,1);

end

