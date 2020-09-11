function [theta_1,theta_2,best_hiddenLayer,best_alpha,best_lambda] = CrossValiditaion(X,y,alphas,lambdas,hidden_counts,max_iter)
% this function finds the optimal values for the neural network
best_alpha = 0;
best_lambda = 0;
best_hiddenLayer = 0;
acc = 0;
theta_1 = []; theta_2 = [];
[~, InputLayerSize] = size(X);
for alpha_t = alphas
    for lambda_t = lambdas
        for layer = hidden_counts
            clc;   
            [J,theta1,theta2] = BackPropagation3(InitializeParam(InputLayerSize, layer),InitializeParam(layer, 2),X,y,max_iter,alpha_t,lambda_t);
            [p] = ForwardPropagation1(theta1,theta2,X);
            accuracy = sum(p==y)/length(y);
            if(accuracy > acc)
                acc = accuracy;
                best_alpha = alpha_t;
                best_lambda = lambda_t;
                best_hiddenLayer = layer;
                theta_1 = theta1;
                theta_2 = theta2;
            end
        end
    end
end
end
