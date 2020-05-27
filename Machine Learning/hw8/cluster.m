% Javier Salazar 1001144647 HW8 K-MEANS CLUSTER PLOT ONLY
clc
clear all % leave these here
%--------input arguments------------
k = [3 6 8 10]; % number of clusters, fixed to ten since usps digits have 10 clusters
trainName = 'USPS_train-2.txt'; % string for importing data
N = 30; % maximum number of iterations to perform mean adjustments
%--------process data------------------------
trainData = readmatrix(trainName); % import data
trainLabels = trainData(:,end) + 1; % change labels to 1-10 for consistancy later on
trainData = trainData(:,1:(end-1)); %  remove label from data matrix
trainData = normalize(trainData, 1, 'zscore'); % normalize based on mean and deviation
testError = zeros(N,length(k)); % for iteration error
for specificCluster=1:length(k)
    %-----------initialize cluster--------------
    rng(7777); % fix random seed to directly observe effect of iterations on accuracy
    clusterLabels = randi([1, k(specificCluster)], length(trainData), 1); % random labels for data, initial labels
    clusters = cell(k(specificCluster),1); % cell to store training points that belong to each cluster
    ind = cell(k(specificCluster), 1); % logical index that determines whether specific points belong to cluster
    % e.g. [0 ... 0 1 0 ... 0] means ith point belongs to kth cluster ind{k}
    meanClusters = zeros(k(specificCluster), size(trainData,2)); % store means for each cluster
    for i = 1:10 % go through each cluster and get initial mean
        ind{i} = (clusterLabels == i); % determine whether points belong to cluster group
        clusters{i} = trainData(ind{i},:); % store points that belong to cluster group i
        meanClusters(i,:) = mean(clusters{i}, 1); % get mean for this group and store
    end
    
    %----------algorithm-------------------------
    distanceMatrix = zeros(length(clusterLabels),k(specificCluster)); % matrix to store distance from each cluster head to each point
    % columns are each cluster head and value is l2 norm from that head
    count = 0; % initialize count
    for n = 1:N % go through N iterations
        for group = 1:k(specificCluster) % for each group, store distance from head to each point row
            distanceMatrix(:,group) = vecnorm(trainData-meanClusters(group,:), 2, 2);
            % store l2 norm of each row (point) to mean head
        end
        [~, newLabels] = min(distanceMatrix, [], 2); % get position of minimum value
        % this is new label that works better than old label
        for i = 1:k(specificCluster) % for each group do the following
            ind{i} = (newLabels == i); % get logical array for what points belong in that group based on new labels
            clusters{i} = trainData(ind{i},:); % store points in cell that belong to new cluster
            meanClusters(i,:) = mean(clusters{i}, 1); % get new mean value
            clusterLabel = mode(trainLabels(ind{i})); % get most common label of points in this cluster
            count = count + sum(( clusterLabel~=trainLabels(ind{i}) )); % sum everywhere there is error between cluster group label and training labels
        end
        testError(n, specificCluster) = count/length(trainLabels); % get error for each iteration
        count = 0; % reset count for each seperate iteration
    end
end
%----------------plotting-----------------
figure
plot(1:1:N,testError(:,1), 'Marker', 'o', 'MarkerSize', 15);
hold on
plot(1:1:N,testError(:,2), 'Marker', 'o', 'MarkerSize', 15);
hold on
plot(1:1:N,testError(:,3), 'Marker', 'o', 'MarkerSize', 15);
hold on
plot(1:1:N,testError(:,4), 'Marker', 'o', 'MarkerSize', 15);
hold off
title('K-Means Clustering On USPS Data As Iteration Number Increases', 'FontSize', 20);
ylabel('Misclassification Error (%)', 'FontSize', 15);
xlabel('Iteration Index (N)', 'FontSize', 15);
legend('3 Clusters', '6 Clusters', '8 Clusters', '10 Clusters', 'FontSize', 15);

