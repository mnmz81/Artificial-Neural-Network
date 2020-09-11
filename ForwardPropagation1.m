function [p] = ForwardPropagation1(Theta1, Theta2, X)
%ForwardPropagation1 employs forward propagation on a 3 layer networks and
% determine the labels of  the inputs 

% Initializations

m = size(X, 1);
X1=[ones(size(X,1),1) X];
z2=X1*Theta1';
a2=sigmoid(z2);
a2=[ones(size(a2,1),1) a2];
z3=a2*Theta2';
a3=sigmoid(z3);
[~,p]=max(a3');
p=p';







