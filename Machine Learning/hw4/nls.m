% Javier Salazar 1001144647 HW4
% This code generates the EPE graph for cross-validation of training data
% using the natural cublic splines model
clc
%------input arguments---------------
trainName = 'Train-1.txt';
knotPoints = 5;
lambdaVector = 0:10^(-4):(50*10^(-4)); % lambda values to test
%--------import data--------------
trainData = readmatrix(trainName);
trainData = sortrows(trainData); % sort data so the removal of knot point during cv only shifts one value
trainData2=trainData; % create dummy matrix where rows will be removed
trainSize = size(trainData,1); % number of training points
%--------knot points generation---------------
firstRange = min(trainData(:,1)); % minimum x value
lastRange = max(trainData(:,1)); % maximum x value
step = (lastRange-firstRange)/(knotPoints+1); % determine step size based on number of knotpoints required
knots = linspace(firstRange+step, lastRange-step, knotPoints); % linearly get theoretical knot point locations
for i=1:knotPoints
    [~, index] = min(abs(trainData(:,1)-knots(i))); % where does the best knot point occur
    knots(i) = trainData(index,1); % store the valid knot point based on training data
end
%------------omega matrix-----------------------
omega = zeros(knotPoints, knotPoints); % initialize omega matrix
syms t % symbolically solve the N" equations
dd_d = cell(1,knotPoints-1); % initialize the di" functions
for k = 1:knotPoints-1 % go through and create symbolic di" functions 
    dd_d{k} = (6*(piecewise(t-knots(k) > 0, t-knots(1), 0)-piecewise(t-knots(knotPoints) > 0, t-knots(knotPoints), 0)))/(knots(knotPoints)-knots(k));
    % must use piecewise instead of max due to symbolic necessity
end
derivatives = cell(1,knotPoints); % intialize derivative functions for N"
derivatives{1}=sym(0);derivatives{2}=sym(0); % N1" and n2" are both zero
for i = 1:knotPoints-2
    derivatives{i+2} = dd_d{i}-dd_d{knotPoints-1}; % define other Ni" functions
end
for i = 1:knotPoints
    for j = 1:knotPoints %go through omega matrix and calculate values
       omega(i,j) = double(int(derivatives{i}*derivatives{j},[firstRange lastRange]));
       % use integration function over whole t domain and solve
       % symbolically 
    end
end
%-------main function-------------
testError = zeros(trainSize, length(lambdaVector)); % matrix for test error based on lambda value and cross validation point left out
for i = 1:trainSize
    testData = trainData(i,:); % what will be considered "test" data
    trainData2(i,:) = []; % remove row from matrix
    for k = 1:length(lambdaVector)
        testError(i,k) = naturalSpline(trainData2, lambdaVector(k), testData, knots, knotPoints, trainSize, omega);
    end
    trainData2=trainData; % redefine dummy matrix so that all data is there again. 
end
testError = mean(testError, 1); % find mean among CV values
plot(lambdaVector,testError); % plot EPE
title('Estimated Prediction Error for Natural Cubic Splines', 'FontSize',20);
xlabel('Lambda Value', 'FontSize', 20);
ylabel('EPE Via Leave-One-Out Cross-Validation', 'FontSize', 20);
%----------cubic spline algorithm-----------------------------
function error = naturalSpline(X, lambda, y, knots, knotPoints, trainSize, omega)
%------N matrix-----------------------
N = zeros(trainSize-1, knotPoints); % for training data
d_y=zeros(1,knotPoints-1); % di values for test data
ny_fun=zeros(1,knotPoints-2); % N values for test data
d = zeros(1,knotPoints-1);n_fun = zeros(1,knotPoints-2); % di values for training data
for i=1:(trainSize-1)
    for k = 1:knotPoints-1
        d(k) = (max((X(i,1)-knots(k))^3,0)-max((X(i,1)-knots(knotPoints))^3,0))/(knots(knotPoints)-knots(k));
        % di values for trainig data
    end
    for j = 1:knotPoints-2
        n_fun(j) = d(j)-d(knotPoints-1); % construct N values for training data
    end
    N(i,:) = [1 X(i,1) n_fun]; % combine all N functions including N1 and N2
end

for k = 1:knotPoints-1
    d_y(k) = (max((y(1)-knots(k))^3,0)-max((y(1)-knots(knotPoints))^3,0))/(knots(knotPoints)-knots(k));
    % di values for testing data
end
for j = 1:knotPoints-2
    ny_fun(j) = d_y(j)-d_y(knotPoints-1);
    % construct N values for test data
end
n_y = [1 y(1) ny_fun]; % create combined N vector with N1 and N2 for test data
theta = (transpose(N)*N + lambda.*omega)\transpose(N)*X(:,2); % get optimized weights for N basis functions
predValue = n_y*theta; % calculate predicted value given test data
error=abs(predValue-y(2)); % error based on measured test data
end
