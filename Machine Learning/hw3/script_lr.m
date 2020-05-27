% Javier Salazar 1001144647 HW3 Logistic Regression
clc
clear all
% main function
% import yeast dataset and do different training data percentages
ErrorYeast100Linear = logRegression(1,'Yeast_train.txt', 'Yeast_test.txt', 1);
ErrorYeast75Linear= logRegression(0.75,'Yeast_train.txt', 'Yeast_test.txt', 1);
ErrorYeast50Linear= logRegression(0.50,'Yeast_train.txt', 'Yeast_test.txt', 1);
ErrorYeast25Linear= logRegression(0.25,'Yeast_train.txt', 'Yeast_test.txt', 1);
ErrorYeast5Linear = logRegression(0.05,'Yeast_train.txt', 'Yeast_test.txt', 1);
% perform quadratic logistic regression
ErrorYeast100Quad = logRegression(1,'Yeast_train.txt', 'Yeast_test.txt', 2);
ErrorYeast75Quad = logRegression(0.75,'Yeast_train.txt', 'Yeast_test.txt', 2);
ErrorYeast50Quad = logRegression(0.5,'Yeast_train.txt', 'Yeast_test.txt', 2);
ErrorYeast25Quad = logRegression(0.25,'Yeast_train.txt', 'Yeast_test.txt', 2);
ErrorYeast5Quad = logRegression(0.05,'Yeast_train.txt', 'Yeast_test.txt', 2);
% satellite data
ErrorSat100Linear = logRegression(1,'Satellite_train.txt', 'Satellite_test.txt', 1);
ErrorSat75Linear = logRegression(0.75,'Satellite_train.txt', 'Satellite_test.txt', 1);
ErrorSat50Linear = logRegression(0.5,'Satellite_train.txt', 'Satellite_test.txt', 1);
ErrorSat25Linear = logRegression(0.25,'Satellite_train.txt', 'Satellite_test.txt', 1);
ErrorSat5Linear = logRegression(0.05,'Satellite_train.txt', 'Satellite_test.txt', 1);
% quadratic logistic regression
ErrorSat100Quad = logRegression(1,'Satellite_train.txt', 'Satellite_test.txt', 2);
ErrorSat75Quad = logRegression(0.75,'Satellite_train.txt', 'Satellite_test.txt', 2);
ErrorSat50Quad = logRegression(0.5,'Satellite_train.txt', 'Satellite_test.txt', 2);
ErrorSat25Quad = logRegression(0.25,'Satellite_train.txt', 'Satellite_test.txt', 2);
ErrorSat5Quad = logRegression(0.05,'Satellite_train.txt', 'Satellite_test.txt', 2);
% plot for order 1 logistic regression
figure
x = [5 25 50 75 100];
group100 = [ErrorYeast100Linear, ErrorSat100Linear];group75=[ErrorYeast75Linear, ErrorSat75Linear];
group50 = [ErrorYeast50Linear, ErrorSat50Linear];group25 = [ErrorYeast25Linear, ErrorSat25Linear];group5 = [ErrorYeast5Linear, ErrorSat5Linear];
bar(x,[group5; group25; group50; group75;group100]);
title('Logistic Regression (Polynomial Order: 1) Misclassification Probability', 'FontSize',20);
xlabel('Data Percentage (%)', 'FontSize', 20);
ylabel('Misclassification Probability', 'FontSize', 20);
legend('Yeast Train Error', 'Yeast Test Error', 'Satellite Train Error', 'Satellite Test Error', 'NumColumns', 1, 'FontSize', 15);
% plot for quadratic logistic regression and both datasets
figure
group100Q = [ErrorYeast100Quad, ErrorSat100Quad];group75Q=[ErrorYeast75Quad, ErrorSat75Quad];
group50Q = [ErrorYeast50Quad, ErrorSat50Quad];group25Q = [ErrorYeast25Quad, ErrorSat25Quad];group5Q = [ErrorYeast5Quad, ErrorSat5Quad];
bar(x,[group5Q; group25Q; group50Q; group75Q;group100Q]);
title('Logistic Regression (Polynomial Order: 2) Misclassification Probability', 'FontSize',20);
xlabel('Data Percentage (%)', 'FontSize', 20);
ylabel('Misclassification Probability', 'FontSize', 20);
legend('Yeast Train Error', 'Yeast Test Error', 'Satellite Train Error', 'Satellite Test Error', 'NumColumns', 1, 'FontSize', 15);

% logistic algorithm
function error = logRegression(dataPercent, trainName, testName, degreePoly)
% input arguments
% data percent e.g. 0.5 for 50% training data
% degreePoly: 1 or 2
% trainName, testName e.g. 'Yeast_test.txt'
lambda = 0.1; % avoid singular matrix issues
s = rng(444);
% import data
countInput = size(readmatrix(trainName), 1); % how much data we will have
countOutput = size(readmatrix(testName), 1);
% uniformly sample data with different percentages
trainData = datasample(readmatrix(trainName), round(dataPercent*countInput), 1, 'Replace', false);
testData = datasample(readmatrix(testName), round(1*countOutput), 1, 'Replace', false);
% set to binary classification. map class 1-10 to 1 and 0
columns = size(trainData,2);
output = trainData(:,columns); class = double(output<=1); trainData(:,columns) = class;
output = testData(:,columns); class = double(output<=1); testData(:,columns) = class;
clear class output
% logistic regression
y = trainData(:,columns); % labels
% indicator matrix with training data
if (degreePoly == 1) % order 1 1 x_1 x_2 ... x_p
    X = [ones(size(trainData,1),1) trainData(:,1:(size(trainData,2))-1)];
    X_test = [ones(size(testData,1),1) testData(:,1:(size(testData,2))-1)];
end
if (degreePoly == 2) % order 2 1 x_1 x_1^2 x_2 x_2^2 ...
    X = zeros(size(trainData,1), 1 + degreePoly*(columns-1));
    X_test = zeros(size(testData,1), 1 + degreePoly*(columns-1));
    for i = 1:size(trainData,1)
        combinedFactors = [trainData(i,1:columns-1); trainData(i,1:columns-1).*trainData(i,1:columns-1)];
        combinedFactors = combinedFactors(:)';
        X(i,:) = [1 combinedFactors];
    end
    for i = 1:size(testData,1)
        combinedFactors = [testData(i,1:columns-1); testData(i,1:columns-1).*testData(i,1:columns-1)];
        combinedFactors = combinedFactors(:)';
        X_test(i,:) = [1 combinedFactors];
    end
end
% initialize coeffecients
b = zeros(1,1 +degreePoly*(columns-1));
% perform newton-ralphson method
temp = 1;
like_old = 1; like = 100; % initialize for first run
% condition is to stop when inv. matrix is ill-conditioned
while (abs(like-like_old) > 1e-2) % stop when cost function doesnt see any benefit
    like_old = like; % replace cost value
    % create beta matrix for indicator matrix
    b_mat = repmat(b, size(X,1), 1);
    % take dot product to generate input for prob. function
    z = dot(b_mat, X, 2);
    % get prob of training data point beloning to class 1
    prob = exp(z)./(exp(z)+ones(length(z),1));
    % generate Weight coefficient matrix
    W = diag(prob.*(1-prob));
    % the matrix that will be inverted and checked for condition number
    temp = transpose(X)*W*X + lambda.*eye(1 + degreePoly*(columns-1),1 + degreePoly*(columns-1));
    % get new coefficients
    b = b + transpose((temp)\transpose(X)*(y-prob));
    % cost function
    like = sum(log(prob));
end
% get misclassification probability for training
trainError = (length(prob)-sum(eq(y,round(prob))))/length(prob);
% get test error
b_mat = repmat(b, size(X_test,1), 1);
z_test = dot(b_mat, X_test, 2);
prob = exp(z_test)./(exp(z_test)+ones(length(z_test),1));
testError = (length(prob)-sum(eq(testData(:,columns),round(prob))))/length(prob);
error = [trainError, testError];
end
