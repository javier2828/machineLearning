% Javier Salazar 1001144647 HW1 Problem 6
% See Terminal for output results 
clc
% import data
x = readmatrix('Tri.txt');
t = readmatrix('Tro.txt', 'ExpectedNumVariables', 1, 'Delimiter', '\n');

% (1) linear regression PART ONE
y = [ones(size(x,1),1) x]; % create data matrix with initial weights
w = (inv(transpose(y)*y))*transpose(y)*t; % find optimal parameters given nice data and output data
error = transpose(t-y*w)*(t-y*w); % compute minimum error

disp('---------PART 1-------------'); % display information
disp('Optimal W Numbers:');
disp(w);
disp('Error Value:');
disp(error);

% (2) REGULARIZED WEIGHTS PART TWO
lambda = 1; % regularization parameter
w_l2 = (inv(lambda*eye(size(y,2)) + transpose(y)*y))*transpose(y)*t; % add lambda to the invertable matrix given number of columns of y
error_l2 = transpose(t-y*w_l2)*(t-y*w_l2); % compute error with regularization added to linear model

disp('-----------PART 2-----------'); % display information
disp('Lambda Value: ');
disp(lambda);
disp('Optimal W Numbers:');
disp(w_l2);
disp('Error Value:');
disp(error_l2);
disp('----------------------');
