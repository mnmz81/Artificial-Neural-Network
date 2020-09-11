function [p] = ForwardPropagation2(theta1, theta2,theta3, X)
%ForwardPropagation2 employs forward propagation on a 4 layer networks and
% determine the labels of  the inputs 

% Initializations
m = size(X, 1);
X1 = [ones(m ,1) X];
z2 = X1*theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1) a2];
z3 = a2*theta2';
a3 = sigmoid(z3);
a3 = [ones(size(a3,1),1) a3];
z4 = a3*theta3';
a4 = sigmoid(z4);

[~,p]=max(a4');
p=p(:);