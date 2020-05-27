% Javier Salazar 1001144647 HW7 Nearest Neighbor
clc
%--------input arguments---------------
trainName = 'USPS_train-1.txt'; % training and test files
testName = 'USPS_test-1.txt';
k = 1:5:50; % vector of k values to get classification rate
%------import & process data--------------
trainData = readmatrix(trainName); % import data
testData = readmatrix(testName);
trainLabels = trainData(:,end); % seperate labels
testLabels = testData(:,end);
trainData = trainData(:,1:end-1); % strp label column
testData = testData(:,1:end-1);
trainData = normalize(trainData, 1, 'zscore'); % normalize based on mean and deviation
testData = normalize(testData, 1, 'zscore');
%----------main function--------------------
classRate = zeros(1,length(k)); % store rates for different k-NN
for l = 1:length(k)
    classRate(l) = bruteNeighbor(trainData, trainLabels, testData, testLabels, k(l)); % go through and get rate based on data
end
figure % plot stuff to make things look nice
plot(k,classRate, '-o','MarkerSize', 10, 'MarkerIndices',1:1:length(k));
title('k-NN Classification Rate Vs. K', 'FontSize', 20);
ylabel('Classification Rate', 'FontSize', 15);
xlabel('Number of Neighbors (k)','FontSize', 15);
%---------nearest neighbor search (bruteforce)---------------
function classificationRate = bruteNeighbor(trainData, trainLabels, testData, testLabels, k)
classificationCount = 0; % count for test points that are correct
for i=1:length(testLabels) % go through all test points
  distanceMatrix = vecnorm(trainData-testData(i,:), 2, 2); % subtract test point from each training point row in matrix
  % then take l2 norm of each row to get distance vector
  [~ , index] = sort(distanceMatrix); % sort based on ascending order and get index
  neighbors = index(1:k); % keep only the first k neighbors
  neighborClasses = trainLabels(neighbors); % et classes of those points
  pointClass = mode(neighborClasses); % select most common class as test class
  if (pointClass == testLabels(i))
      classificationCount = classificationCount + 1; % if k-NN label matches true label then correct
  end
end
classificationRate = classificationCount/length(testLabels); % get percentage
end