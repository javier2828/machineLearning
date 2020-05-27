% Javier Salazar 1001144647 HW3
clc
clear all
% main function
% get training and test error vector from lda function
% Yeast dataset with various data percantage for training
ErrorYeast100 = lda(1, 'Yeast_train.txt', 'Yeast_test.txt');
ErrorYeast50 = lda(0.5, 'Yeast_train.txt', 'Yeast_test.txt');
ErrorYeast25 = lda(0.25, 'Yeast_train.txt', 'Yeast_test.txt');
ErrorYeast75 = lda(0.75, 'Yeast_train.txt', 'Yeast_test.txt');
ErrorYeast5 = lda(0.05, 'Yeast_train.txt', 'Yeast_test.txt');
% Satellite Dataset
ErrorSat100 = lda(1, 'Satellite_train.txt', 'Satellite_test.txt');
ErrorSat75 = lda(0.75, 'Satellite_train.txt', 'Satellite_test.txt');
ErrorSat50 = lda(0.5, 'Satellite_train.txt', 'Satellite_test.txt');
ErrorSat25 = lda(0.25, 'Satellite_train.txt', 'Satellite_test.txt');
ErrorSat5 = lda(0.05, 'Satellite_train.txt', 'Satellite_test.txt');

% plot bar graph
figure
x = [5 25 50 75 100];
group100 = [ErrorYeast100, ErrorSat100];group75=[ErrorYeast75, ErrorSat75];
group50 = [ErrorYeast50, ErrorSat50];group25 = [ErrorYeast25, ErrorSat25];group5 = [ErrorYeast5, ErrorSat5];
bar(x,[group5; group25; group50; group75;group100]);
title('Linear Discriminant Analysis (LDA) Misclassification Probability', 'FontSize',20);
xlabel('Data Percentage (%)', 'FontSize', 20);
ylabel('Misclassification Probability', 'FontSize', 20);
legend('Yeast Train Error', 'Yeast Test Error', 'Satellite Train Error', 'Satellite Test Error', 'NumColumns', 1, 'FontSize', 15);

% lda algorithm
function error = lda(dataPercent, trainName, testName)
% input arguments
% dataPercent e.g. 0.5 for 50% training data
% trainName,testName e.g. = 'Yeast_train.txt'
lambda = 1e-12; % prevents singular matrix inversion for in-class covarience matrix
countTrain = size(readmatrix(trainName),1); % how much data we have
countTest = size(readmatrix(testName),1);
% sample data uniformly
trainData = datasample(readmatrix(trainName), round(dataPercent*countTrain), 1, 'Replace', false);
testData = datasample(readmatrix(testName), round(1*countTest), 1, 'Replace', false);
% set to binary classification (class 1-10 mapped to class 1 and 0)
columns = size(trainData,2);
output = trainData(:,columns); class = double(output<=1); trainData(:,columns) = class;
output = testData(:,columns); class = double(output<=1); testData(:,columns) = class;
clear class output
%seperate the two classes of data from training data
X = trainData(:, 1:(columns-1));
X_class1   = trainData( trainData(:,columns)==1, :);X_class1 = X_class1(:,1:columns-1);
X_class2   = trainData( trainData(:,columns)==0, :);X_class2 = X_class2(:,1:columns-1);
labels = trainData(:,columns);
% get hyperparameters of model 
mean1 = mean(X_class1, 1); % mean feature values
mean2 = mean(X_class2, 1);
prior1 = size(X_class1, 1)/size(trainData,1); % how many points of class 1 acquired
prior2 = size(X_class2, 1)/size(trainData,1);
cov1 = cov(X_class1); % in-class covariance matrices
cov2 = cov(X_class2);
covTotal = (cov1+cov2)./(size(trainData,1)-2); % lda assumes same matrix so combine them
covTotal = covTotal + lambda.*eye(size(covTotal)); % prevent singular matrix issues
% calculate linear discrimenent functions
delta1 = X*inv(covTotal)*transpose(mean1) - 0.5.*mean1*inv(covTotal)*transpose(mean1) + log(prior1);
delta2 = X*inv(covTotal)*transpose(mean2) - 0.5.*mean2*inv(covTotal)*transpose(mean2) + log(prior2);
delta = [delta1 delta2];
% delta is for training points to belong to class 1 and 0
clear delta1 delta2
% get label information
[~, trainLabel] = max(delta, [], 2);
trainLabel = (trainLabel<=1); % get likely labels given maximum LDF
% training error probability for misclassification
trainError = (length(trainLabel)-sum(eq(labels,trainLabel)))/length(trainLabel);
% testing error calculation
X_test = testData(:, 1:(columns-1));
labels_test = testData(:,columns);
delta1 = X_test*inv(covTotal)*transpose(mean1) - 0.5.*mean1*inv(covTotal)*transpose(mean1) + log(prior1);
delta2 = X_test*inv(covTotal)*transpose(mean2) - 0.5.*mean2*inv(covTotal)*transpose(mean2) + log(prior2);
delta_test = [delta1 delta2];
clear delta1 delta2
[~, testLabel] = max(delta_test, [], 2);
testLabel = (testLabel<=1);
testError = (length(testLabel)-sum(eq(labels_test,testLabel)))/length(testLabel);
error = [trainError, testError];
end