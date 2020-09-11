function [J,theta1,theta2,theta3] = BackPropagation4(theta1, theta2,theta3, X,y,max_iter, alpha,Lambda)
% BP BackPropagation for training a neural network
% Input arguments
% Theta1 - matrix of parameters (weights)  between the input and the first hidden layer
% Theta2 - matrix of parameters (weights)  between the hidden layer and the output layer (or
% another hidden layer)
% X - input matrix
% y - labels of the input examples
% max_iter - maximum number of iterations (epochs).
% alpha - learning coefficient.
% Lambda - regularization coefficient.
%
% Output arguments
% J - the cost function
% Theta1 - updated weight matrix between the input and the first hidden layer
% Theta2 - updated weight matrix between the hidden layer and the output layer (or
% second hidden layer)
%
% Usage:[J,Theta1,Theta2] = bp(Theta1, Theta2, X,y,max_iter, alpha,Lambda)
%

% Initalization
if nargin<8, Lambda=0; end
if nargin<7, alpha=0.1; end
if nargin<6, max_iter=1000; end

m = size(X,1);
num_outputs = size(theta3, 1);
delta4 = zeros(num_outputs,1);   % delta's for the output layer   
delta3=zeros(size(theta2,1),1); % delta's for the hidden layer 2    
delta2=zeros(size(theta1,1),1); % delta's for the hidden layer 1
ybin=zeros(size(delta4)); 
p = zeros(size(X, 1), 1);
J=0;
for q=1:max_iter
    J=0;
    dTheta3 = zeros(size(theta3));
    dTheta2 = zeros(size(theta2));
    dTheta1 = zeros(size(theta1));
    theta3_grad = zeros(size(dTheta3));
    theta2_grad = zeros(size(dTheta2));
    theta1_grad = zeros(size(dTheta1));
    r=randperm(size(X,1));
% feed forward 
    for k=1:m
        X1 = X(r(k),:);
        X1 = [1 X1];
        z2 = X1*theta1';
        a2 = sigmoid(z2);
        a2 = [1 a2];
        z3 = a2*theta2';
        a3 = sigmoid(z3);
        a3 = [1 a3];
        z4 = a3*theta3';
        a4 = sigmoid(z4);
% Backward propagation
        % ---------------------
        ybin=zeros(size(a4));
        if y(r(k))==0, y(r(k))=2; end
        ybin(y(r(k)))=1; % Assigning 1 to the binary digit according to the class (digit). ybin(2)=1 for '0'.
        J = J+1/m*(-ybin*log(a4')-(1-ybin)*log(1-a4')); % Maximum likelihood cost function
        delta4 = (a4-ybin)';
        delta3 = (theta3'*delta4).*(a3'.*(1-a3')); 
        delta2 = (theta2'*delta3(2:end)).*(a2'.*(1-a2'));
        dTheta3 = dTheta3 + delta4*a3;
        dTheta2 = dTheta2 + delta3(2:end)*a2;
        dTheta1 = dTheta1 + delta2(2:end)*X1;
    end
    % updating J with the regularization cost
    J=J+(Lambda/(2*m))*(sum(sum((theta1(:,2:end)).^2))...
        + sum(sum((theta2(:,2:end)).^2)) +sum(sum((theta3(:,2:end)).^2)));
    if mod(q,1000)==0
        fprintf(['\n Cost function J = %f in iteration %d with Lambda = %.2f & alpha = %d \n'],J,q,Lambda,alpha);
        pause(0.0005)
    end
    theta3_grad = 1/m*dTheta3;
    theta3_grad(:,2:end) = theta3_grad(:,2:end)+Lambda/m*theta3(:,2:end);
    theta2_grad = 1/m*dTheta2;
    theta2_grad(:,2:end) = theta2_grad(:,2:end)+Lambda/m*theta2(:,2:end);
    theta1_grad = 1/m*dTheta1;
    theta1_grad(:,2:end) = theta1_grad(:,2:end)+Lambda/m*theta1(:,2:end);
    theta3 = theta3-alpha*theta3_grad;
    theta2 = theta2-alpha*theta2_grad;
    theta1 = theta1-alpha*theta1_grad; % Updating the parameters (weights)
    if mod(q,100)==0
        p = ForwardPropagation2(theta1, theta2,theta3, X);
        fprintf('\n Network Accuracy for Training Set with (%d,%d) nodes at hidden layers: %f \n'...
            ,size(theta1,1),size(theta2,1), sum(p == y)/m * 100);
    end 
end
end