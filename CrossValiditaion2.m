function [theta_1,theta_2,theta_3,hiddenLayer1,hiddenLayer2,alpha,lambda] = CrossValiditaion2(X,y,alpha_vec,lambda_vec,hidden1_vec,hidden2_vec,max_iter)
% this function finds the optimal values for the neural network
alpha = 0; 
lambda = 0;
hiddenLayer1 = 0;
hiddenLayer2 = 0;
acc = 0;
theta_1 = []; theta_2 = []; theta_3=[];
[~, InputLayerSize] = size(X);
for alphas = alpha_vec
    for lambdas = lambda_vec
        for layer1 = hidden1_vec
            for layer2 = hidden2_vec
                clc;
                [J,theta1,theta2,theta3] = BackPropagation4(InitializeParam(InputLayerSize, layer1),...
                InitializeParam(layer1, layer2),InitializeParam(layer2, 2),X,y,max_iter,alphas,lambdas);
                p = ForwardPropagation2(theta1,theta2,theta3,X);
                accuracy = sum(p==y)/length(y);
                if(accuracy > acc) 
                    acc = accuracy;
                    alpha = alphas;
                    lambda = lambdas;
                    hiddenLayer1 = layer1;
                    hiddenLayer2 = layer2;
                    theta_1 = theta1;
                    theta_2 = theta2;
                    theta_3 = theta3;
                end
            end 
        end
    end
end
end
