function [ y ] = propSig(p,q,W,s)
% Computes the propagation of a signal s from layer p to layer q

if (size(W,1)~=size(W,2))
    disp('dimension Mismatch'); 
end
if (size(W,1)~=length(s))
    disp('dimension Mismatch');
end


g = @nonLin;
y = s;
for i=p:q-1
    temp = g(W(:,:,i)*y);
    y=temp;
end


end

