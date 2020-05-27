% Javier Salazar 1001144647 Problem 1
clc
%--------input arguments-----------------
trainingFile = 'Train.txt'; % assuming files in same directory
testFile = 'Test.txt';
%--------------main----------------------------
trainData = readmatrix(trainingFile); % import data
testData = readmatrix(testFile);
% these functions will generate 5 graphs based on percentage of train data
% datasample(.) samples data uniformly with no replacement
generatePlot(datasample(trainData, 400, 1, 'Replace', false), testData, '100%')
generatePlot(datasample(trainData, 200, 1, 'Replace', false), testData, '50%')
generatePlot(datasample(trainData, 40, 1, 'Replace', false), testData, '10%')
generatePlot(datasample(trainData, 8, 1, 'Replace', false), testData, '2%')
%----------generate graph for X%% data--------------
function [] = generatePlot(trainData, testData, dataSample)
regVec = [0, 1, 5]; % the reg term values in a vector
figure
for i=1:5 % order of polynomial
    for j=1:3 % go through reg. vector
        [trainError, testError] = polyRegress(trainData, testData, i, regVec(j)); % get train and test error given lambda and order of poly
        if (j == 1) % for lambda=1 give the points this kind of attributes (e.g. red)
            scatter(i, trainError, 72*2, 'red', 'd')
            hold on
            scatter(i, testError , 72*2, 'red', 'x')
            hold on
        end
        if (j == 2) % likewise have different shapes for train and test data
            scatter(i, trainError, 72*2, 'blue', 'd')
            hold on
            scatter(i, testError , 72*2, 'blue', 'x')
            hold on
        end
        if (j == 3)
            scatter(i, trainError, 72*2, 'green', 'd')
            hold on
            scatter(i, testError , 72*2, 'green', 'x')
            hold on
        end
    end
end
% cleaning up graphs and changing to semilogy scale for better graphs
title(['Regularized Multivariate Polynomial Regression Error Results (',dataSample,' Data)'], 'FontSize', 20);
xlabel('Polynomial Order', 'FontSize', 15);
ylabel('L2 Error (Log10 Scale)', 'FontSize', 15);
set(gca,'YScale','log');
legend({'Train Error (Lambda = 0)','Test Error (Lambda = 0)','Train Error (Lambda = 1)','Test Error (Lambda = 1)','Train Error (Lambda = 5)' , 'Test Error (Lambda = 5)'},'FontSize', 18);
hold off
end

%--------polynomial regression------------------
function [errorTrain, errorTest] = polyRegress(trainData, testData, orderPoly, lambda)
% initialize design matrix since it will be filled
Design = zeros(size(trainData,1), (orderPoly+1)*(orderPoly+2)*0.5);
% go through each row in matrix and fill based on lookup function
for i = 1:size(Design,1)
    Design(i,:) = lookup(trainData(i,1), trainData(i,2),orderPoly);
end
% calculate weights based on HW1 method with regularization value
weights = inv(lambda*eye(size(Design, 2)) + transpose(Design)*Design)*transpose(Design)*trainData(:,3);
% return training error based on HW1
errorTrain = transpose(trainData(:,3)-Design*weights)*(trainData(:,3)-Design*weights);
% create design matrix for testing data to get error
DesignTest = zeros(size(testData,1), (orderPoly+1)*(orderPoly+2)*0.5);
% go through each row based on test data
for i = 1:size(DesignTest,1)
    DesignTest(i,:) = lookup(testData(i,1), testData(i,2),orderPoly);
end
% get error based on new design matrix and training weight vector
errorTest = transpose(testData(:,3)-DesignTest*weights)*(testData(:,3)-DesignTest*weights);
end
%----------design matrix row for data points--------------------
function row = lookup(x, y, orderPoly) % return row in design matrix for ith data sample
A = [];B = [];C = [];D=[];E=[]; % initialize empty vectors
for i=1:orderPoly
    if (i == 1)
        A = [1 x y]; % coefficients related to order 1
    end
    if (i == 2)
        B = [x*y x^2 y^2]; % coeffecients needed for second order in addition to 1
    end
    if (i == 3)
        C = [y*x^2 x*y^2 x^3 y^3]; % repeat etc..
    end
    if (i == 4)
        D = [(x^2)*(y^2) (x^3)*(y^1) (x^1)*(y^3) x^4 y^4];
    end
    if (i == 5)
        E = [(x^3)*(y^2) (x^2)*(y^3) (x^1)*(y^4) (x^4)*(y^1) x^5 y^5];
    end
end
row = [A B C D E]; % combine all to get the final row 
end
