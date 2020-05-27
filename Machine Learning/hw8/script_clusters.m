% Javier Salazar 1001144647 HW8 PCA + K-MEANS CLUSTERING
clc
clear all % leave these here
%------------input arguments------------
trainName = 'USPS_train-2.txt'; % string for importing data
k = [3 6 8 10]; % fixed number of cluster groups since USPS digits have 10 classes
N = 30; % fixed number of iterations for K-Means Clustering
pcaComponents = 16; % maximum number of pca components to report error
%--------process data------------------------
trainData = readmatrix(trainName); % import data
trainLabels = trainData(:,end) + 1; % get point labels and change to 1-10 for consistancy
trainData = trainData(:,1:(end-1)); % strip labels from data matrix
trainData = normalize(trainData, 1, 'zscore'); % normalize based on mean and deviation
%--------begin pca + clustering----------------
testError = zeros(length(k), pcaComponents); % empty array for classification error
for specificCluster = 1:length(k)
    for comp=1:pcaComponents % go through for each component number
        %---------pca-------------------------------
        sigmaMatrix = (1/length(trainData))*transpose(trainData)*trainData; % covariance matrix for all variables
        [V, ~] = eig(sigmaMatrix); % eigenvalue decomposition f covariance matrix
        pcaMatrix = zeros(length(trainData), comp); % new lower space data
        for i = 1:comp % go through and store new component values based on however many comp. vectors we are using
            pcaMatrix(:,i) = trainData*V(:,i); % map to new space using component vectors found
            % pcaMatrix is compressed new space
        end
        %--------clustering algorithm-----------------------
        %-----------initialize cluster--------------
        rng(7777); % fix random seed to directly observe effect of iterations on accuracy
        clusterLabels = randi([1, k(specificCluster)], length(pcaMatrix), 1); % random labels for data, initial labels
        clusters = cell(k(specificCluster),1); % cell to store training points that belong to each cluster
        ind = cell(k(specificCluster), 1); % logical index that determines whether specific points belong to cluster
        % e.g. [0 ... 0 1 0 ... 0] means ith point belongs to kth cluster ind{k}
        meanClusters = zeros(k(specificCluster), size(pcaMatrix,2)); % store means for each cluster
        for i = 1:10 % go through each cluster and get initial mean
            ind{i} = (clusterLabels == i); % determine whether points belong to cluster group
            clusters{i} = pcaMatrix(ind{i},:); % store points that belong to cluster group i
            meanClusters(i,:) = mean(clusters{i}, 1); % get mean for this group and store
        end
        %----------cluster-------------------------
        distanceMatrix = zeros(length(clusterLabels),k(specificCluster)); % matrix to store distance from each cluster head to each point
        count = 0; % initialize count and reset for each component count
        for n = 1:N % go through N iterations
            for group = 1:k(specificCluster) % for each group, store distance from head to each point row
                distanceMatrix(:,group) = vecnorm(pcaMatrix-meanClusters(group,:), 2, 2);
                % store l2 norm of each row (point) to mean head
            end
            [~, newLabels] = min(distanceMatrix, [], 2); % get position of minimum value
            % this is new label that works better than old label
            for i = 1:k(specificCluster) % for each group do the following
                ind{i} = (newLabels == i); % get logical array for what points belong in that group based on new labels
                clusters{i} = pcaMatrix(ind{i},:); % store points in cell that belong to new cluster
                meanClusters(i,:) = mean(clusters{i}, 1); % get new mean value
                if (n == N) % get error during last iteration
                    clusterLabel = mode(trainLabels(ind{i})); % get most common label of points in this cluster
                    count = count + sum(( clusterLabel~=trainLabels(ind{i}) )); % sum everywhere there is error between cluster group label and training labels
                end
            end
        end
        testError(specificCluster,comp) = count/length(trainLabels); % get error for however many components used
    end
end
%------------plotting----------------------
figure
plot(1:1:pcaComponents,testError(1,:), 'Marker', 'o', 'MarkerSize', 15);
hold on
plot(1:1:pcaComponents,testError(2,:), 'Marker', 'o', 'MarkerSize', 15);
hold on
plot(1:1:pcaComponents,testError(3,:), 'Marker', 'o', 'MarkerSize', 15);
hold on
plot(1:1:pcaComponents,testError(4,:), 'Marker', 'o', 'MarkerSize', 15);
hold off
title('Performing PCA + K-Means Clustering On USPS Data (Fixed N = 30 K-Means Iterations)', 'fontSize', 20);
ylabel('Misclassification Error (%)', 'fontSize', 15);
xlabel('k principal components', 'fontSize', 15);
legend('3 Clusters', '6 Clusters', '8 Clusters', '10 Clusters', 'FontSize', 15);
