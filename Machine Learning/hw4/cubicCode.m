% Javier Salazar 1001144647 HW4
% This code will generate plots to compare training data and predictor
% model of natural cubic splines
clc
%------input arguments---------------
trainName = 'Train-1.txt'; % training data
lambda = 0; % regularization parameter
knotPoints = 5;
%--------import data--------------
trainData = readmatrix(trainName);
%--------knot points generation---------------
firstRange = min(trainData(:,1)); % minimum and maximum x value range for training data
lastRange = max(trainData(:,1));
step = (lastRange-firstRange)/(knotPoints+1); % calculate step size based on amount of knot points
knots = linspace(firstRange+step, lastRange-step, knotPoints); % uniformly generate knot vector (theoretica values)
for i=1:knotPoints
    [~, index] = min(abs(trainData(:,1)-knots(i))); % find index of closest knot point
    knots(i) = trainData(index,1); % store experimental knot point closest to theory point
end
%------N matrix generation-----------------------
N = zeros(size(trainData,1), knotPoints); % initialize N matri
d = zeros(1,knotPoints-1); % di values for single row
n_fun = zeros(1,knotPoints-2); % N values for single row
for i=1:size(N,1)
    for k = 1:knotPoints-1
        d(k) = (max((trainData(i,1)-knots(k))^3,0)-max((trainData(i,1)-knots(knotPoints))^3,0))/(knots(knotPoints)-knots(k));
        % calculate di values for single row and knot points
    end
    for j = 1:knotPoints-2
        n_fun(j) = d(j)-d(knotPoints-1);
        % calculate n function for n3 up to knot point N
    end
    N(i,:) = [1 trainData(i,1) n_fun];
    % create final N vector with n1 and n2. store in N matrix
end
%------------omega matrix generation-----------------------
omega = zeros(size(N,2), size(N,2)); % intialize N matrix
syms t % create symbolic variable t
dd_d = cell(1,knotPoints-1); % define "array" to store symbolic di" functions
for k = 1:knotPoints-1
    dd_d{k} = (6*(piecewise(t-knots(k) > 0, t-knots(1), 0)-piecewise(t-knots(knotPoints) > 0, t-knots(knotPoints), 0)))/(knots(knotPoints)-knots(k));
    % store symbolic di" functions as defined in lecture
end
derivatives = cell(1,knotPoints); % intialize symbolic array
derivatives{1}=sym(0);derivatives{2}=sym(0); % N1" and N2" are zero.
for i = 1:knotPoints-2
    derivatives{i+2} = dd_d{i}-dd_d{knotPoints-1};
    % store symbolic Ni" functions
end
for i = 1:size(N,2)
    for j = 1:size(N,2)
       omega(i,j) = double(int(derivatives{i}*derivatives{j},[firstRange lastRange]));
       % calculate Nij values symbolically using integration function and
       % the whole t domain
    end
end
%-------------theta parameter------------------
theta = (transpose(N)*N + lambda.*omega)\transpose(N)*trainData(:,2);
% get optimized weights for the Ni basis functions
predValues = N*theta; % predictor values based on model
S = trace(N*((transpose(N)*N + lambda.*omega)\transpose(N)));
% trace of the S matrix that gives degrees of freedom
%--------plot-----------------------------------
figure
scatter(trainData(:,1), trainData(:,2),36, 'blue');
hold on
scatter(trainData(:,1), predValues, 36, 'red');
% plot training data and predictor model
title(['Training Data and Predictor Model (Lambda = ',num2str(lambda), ', knotPoints = ',num2str(knotPoints),')'],'FontSize', 20);
ylabel('Y Value','FontSize',20);
xlabel('X Value','FontSize', 20);
legend('Training Data', 'Predictor Values','FontSize', 20);
annotation('textbox', [0.45, 0.6, 0.1, 0.1], 'String', strcat('Degrees of Freedom: ',num2str(S)));
% include degrees of freedoms calculation in the graph
hold off

