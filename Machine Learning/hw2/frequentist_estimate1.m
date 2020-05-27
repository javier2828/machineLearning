% Javier Salazar 1001144647 Problem 2
clc
%------generate string---------------------------
randomNumbers = rand(5000, 1); % uniform [0,1] random numbers
randomNumbers = heaviside(randomNumbers - 0.1) + 97; % convert to binary signal and shift
string = char(randomNumbers); % convert 97 --> a and 98 --> b
%------frequentist analysis----------------------
count = 0;
for i = 1:5000 % go through string and count
    if (string(i) == 'a') % calculate number of a's in string
        count = count + 1;
    end
end
prob = count/5000; % get probability of occurance
%-------display results-----------
disp('------------------------');
disp('Theoretical P(c=''a''):');
disp('      0.1');
disp('Frequentist P(c=''a''):');
disp(prob);
disp('------------------------')