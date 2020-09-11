clear;clc;
%% A
load face_train.mat
ytrain(ytrain==-1) = 2; % change -1 to 2 in labels
%% B
figure(1);ShowImage(Xtrain(1,:)); % to show face
figure(2);ShowImage(Xtrain(5000,:));% to show not face 
%% C
[m,n] = size(Xtrain);  

max_iter = 1500;
alpha = [3];
lambda = [0]; 
hiddenLayerSize = [12];
secondLayerSize = [12];

[theta_1,theta_2,theta_3,hiddenLayer1,hiddenLayer2,alpha,lambda] = CrossValiditaion2(Xtrain,ytrain,alpha,lambda,...
    hiddenLayerSize,secondLayerSize,max_iter);
fprintf('the best alpha is %f \n', alpha);
fprintf('the best lambda is %f \n', lambda);
fprintf('the best hiddenLayer1 is %f \n', hiddenLayer1);
fprintf('the best hiddenLayer2 is %f \n', hiddenLayer2);
%% D
load face_test.mat

ytest(ytest==-1) = 2; % change -1 to 2 in labels

p1 = ForwardPropagation2(theta_1,theta_2,theta_3,Xtest);
fprintf('Precision for two hidden layer is %f', sum(ytest==p1)/length(ytest) *100);

