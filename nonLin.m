function [ y yp ] = nonLin( x )
% Nonlinearity in a neural network

beta =1; %Scaling param in nonlinearity

%%Linear function
%  y = beta*x;
%  yp = beta*ones(size(x));



%% tanh function
y= tanh(beta*x);
%derivative of y
yp = beta*(1-y.^2);


%% Logistic function
% y=1./(1+exp(-2*beta*x))
% yp = 2*beta*y.*(1-y);
end

