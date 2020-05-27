% Javier Salazar 1001144647 Problem 3
clc
%-------inputs---------------------------------
lengthString = 5000; % how many random samples
%----------bayesian analysis-----------------------
rng('shuffle') % fixes random generation so only one set of data observed in likelihood function
% random seed will change each time 'Run'is pressed but stays constant in
% likelihood calculation
prior = [0.9, 0.04, 0.03, 0.02, 0.01]; % prior distribution given by problem
% likelihood calculated using frequentist approach
likelihood = [like(0.1, lengthString), like(0.3, lengthString), like(0.5, lengthString), like(0.7, lengthString), like(0.9, lengthString)];
% probability of entire set to scale the posterior
probS = dot(prior,likelihood);
post_01 = (likelihood(1)*prior(1))/probS; % get posterior values for each case
post_03 = (likelihood(2)*prior(2))/probS;
post_05 = (likelihood(3)*prior(3))/probS;
post_07 = (likelihood(4)*prior(4))/probS;
post_09 = (likelihood(5)*prior(5))/probS;
post_m = post_01 + post_03 + post_05 + post_07 + post_09;
%---------print results---------------------------
disp('------------------------');
disp(['P( m = 0.1 | S) = ', num2str(post_01)]);
disp(['P( m = 0.3 | S) = ', num2str(post_03)]);
disp(['P( m = 0.5 | S) = ', num2str(post_05)]);
disp(['P( m = 0.7 | S) = ', num2str(post_07)]);
disp(['P( m = 0.9 | S) = ', num2str(post_09)]);
disp(['P( c = ''a'' | S) = ', num2str(post_m)]);
disp('------------------------')

%--------likelihood function-----------------------
function prob = like(m,lengthString)
%------generate string---------------------------
randomNumbers = rand(lengthString, 1); % uniform [0,1] random numbers
randomNumbers = heaviside(randomNumbers - m) + 97; % convert to binary signal and shift
string = char(randomNumbers); % convert 97 --> a and 98 --> b
%------frequentist analysis----------------------
count = 0;
for i = 1:lengthString % go through string and count
    if (string(i) == 'a') % calculate number of a's in string
        count = count + 1;
    end
end
prob = count/5000; % get probability of occurance
end