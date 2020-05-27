% Javier Salazar 1001144647
% HW6 Neural Network (Multi-layer Perceptron)
clc
%--------input arguments---------------------------
trainName = 'USPS_train.txt'; % training and test files
testName = 'USPS_test.txt';
hiddenLayers = 1; % excludingnput and output layers
perceptronCount = 15; % hidden layers perceptrons per layer
epochs = 100; % stop after this many iterations
classes = 10; % output classes in total
% good specs: 100 epochs, 14 perceptrons per layer, 1 layers
%---------import and process data-------------------------
neuralLayers = hiddenLayers + 2; % total number of layers
trainData = readmatrix(trainName); % import data
testData = readmatrix(testName);
features = size(trainData,2); % how many input features we have
trainLabels = trainData(:,end) + 1; testLabels = testData(:,end) + 1; % get labels and offset s.t. digit 0 = class 1
trainData = trainData(:,1:(end-1)); testData = testData(:,1:(end-1)); % strip labels from data
trainData = normalize(trainData, 1, 'range'); % normalize based on maximum feature value per column
testData = normalize(testData, 1, 'range');
trainData = [ones(size(trainData,1),1) trainData]; % add bias input feature to data
testData = [ones(size(testData,1),1) testData];
T_train = zeros(size(trainLabels,1), classes); % create sparse matrix with class data 
T_test = zeros(size(testLabels,1), classes); % e.g. t1 = [0 0 ... 0 1 0 ... 0] for first label of class C_k
for i = 1:size(trainLabels, 1) % algorithm to create label matrix for each label
    T_train(i, trainLabels(i)) = 1; 
end
for i = 1:size(testLabels, 1)
    T_test(i, testLabels(i)) = 1;  
end
%--------initialize parameters------------------------
rng(55555); % fix seed generation to directly observe effects of perceptrons and layers only
% also to reproduce results shown in answers.pdf if needed
perceptronTotal = (neuralLayers-2)*perceptronCount +classes+features; % total perceptrons including input and output layers
weight_x1 = -0.05; weight_x2 = 0.05; % domain range to randomly generate initial weight matrix
weights = weight_x1 + (weight_x2-(weight_x1)).*rand(perceptronTotal,perceptronTotal); % use uniform distribution
index = cell(1,neuralLayers); index{1} = 1:1:features; % create layer map that specifies where each perceptron belongs
for i = 2:neuralLayers-1 % fill index cell array with the perceptron values
index{i} = features+1+(i-2)*perceptronCount:1:features+perceptronCount+(i-2)*perceptronCount;
end
index{neuralLayers} = perceptronTotal-classes+1:1:perceptronTotal;
stepSize = 10; % learning rate to update weights
a = zeros(1, perceptronTotal); % linear combination of inputs for each perceptron
delta = zeros(1, perceptronTotal); % delta values to determine weight updates
z = zeros(1, perceptronTotal); % output of perceptron after sigmoid activation
bias = weight_x1 + (weight_x2-(weight_x1)).*rand(1,perceptronTotal); % bias weights intialization
%-----------multi-layer perceptron------------------------
disp('--------Training Phase-----------');
for i = 1:epochs % iterations
    for n = 1:size(trainLabels,1) % go through each input data
        x = trainData(n,:); % input point
        t = T_train(n,:); % input label
        %-----go forward-----------------
        z(index{1}) = x; % input features belong to input perceptrons
        for l = 2:neuralLayers % go through remainding layers
            currentLayer = index{l}; % current layer perceptrons
            outputClass = 1; % indexing integer
            for j = 1:length(currentLayer) % go through all perceptrons in layer
                a(currentLayer(j)) = dot(weights(currentLayer(j),:), z) + bias(currentLayer(j)); % get linear combination of inputs plus bias weight*1
                z(currentLayer(j)) = (1+exp(-a(currentLayer(j))))^-1; % output of perceptron after sigmoid
                if (l == neuralLayers) % for last layer also perform delta calculation since error is directly observed
                    delta(currentLayer(j)) = (z(currentLayer(j))-t(outputClass))*z(currentLayer(j))*(1-z(currentLayer(j)));
                    % e.g. data is class 2 but class 1 = 0.2 so error seen
                    outputClass = outputClass + 1; % delta values for each output perceptron
                end
            end
        end
        %---------go backward (backproprogation)---------------------------------------
        for l = neuralLayers-1:-1:2 % go backwards for hidden layers
            currentLayer = index{l}; % current layer perceptrons
            futureLayer = index{l+1}; % perceptrons on adjacent future layer
            for j = 1:length(currentLayer) % go through all perceptrons and get derivative values
                delta(currentLayer(j)) =( dot(delta(futureLayer), weights(futureLayer,j)) )*z(currentLayer(j))*(1-z(currentLayer(j))); 
            end
        end
        %---------update weights------------------------------
        for l = 2:1:neuralLayers % order does not matter here
            currentLayer = index{l};
            pastLayer = index{l-1}; % perceptrons located in respective layers
            for j = 1:length(currentLayer) % go through current layer perceptrons
                for k = 1:length(pastLayer) % update weights based on delta values and output perceptron values from past layer
                    weights(currentLayer(j), pastLayer(k)) = weights(currentLayer(j), pastLayer(k)) - stepSize*delta(currentLayer(j))*z(pastLayer(k));
                end
                bias(currentLayer(j)) = bias(currentLayer(j)) - stepSize*delta(currentLayer(j))*1; % update bias weights
            end
        end
        %---------end for single training data----------------
    end
    stepSize = stepSize*0.98; % decrease learning rate for each epoch
    disp(['Epoch: ',num2str(i),'/',num2str(epochs)]);
    %----------end for all iterations----------------------
end
%-------------testing phase------------------------------------
disp('--------Testing Phase----------');
a = zeros(1, perceptronTotal); z = zeros(1, perceptronTotal); % initialize test vectors
z_test = zeros( size(testLabels,1), classes);
for n = 1:size(testLabels,1) % go through each 
    x = testData(n,:); % test point vector
    t = T_test(n,:); % test label vector
    %-----go forward-----------------
    z(index{1}) = x; % perceptron output for input perceptrons
    for l = 2:neuralLayers % go through remaining layers
        currentLayer = index{l}; % perceptrons belonging to current layer
        outputClass = 1; % counting integer for output perceptrons
        for j = 1:length(currentLayer) % go through all perceptrons
            a(currentLayer(j)) = dot(weights(currentLayer(j),:), z) + bias(currentLayer(j)); % get linear combination of inputs to perceptron and weights plus bias weights*1
            z(currentLayer(j)) = (1+exp(-a(currentLayer(j))))^-1; % apply sigmoid activation to get output perceptron value
        end
    end
    z_test(n,:) = z(perceptronTotal-classes+1:1:perceptronTotal); % store output layer perceptrons as class information
    %-------test point evaluated
end
%-----------classification calculation and display---------------------
[~, testLabelsPredicted] = max(z_test, [], 2); % get maximum location for each row (test point)
testError = (length(testLabelsPredicted)-sum(eq(testLabels,testLabelsPredicted)))/length(testLabelsPredicted); % misclassification rate based on incorrect predicted classes
classificationRate = 1-testError; % success rate
for n = 1:size(testLabels,1) % go through all test points and write true and predicted class in terminal
    disp(['Datum ID: ', num2str(n),'     Predicted Class: ',num2str(testLabelsPredicted(n)-1),'     True Class: ',num2str(testLabels(n)-1)]);
end
disp(['Classification Accuracy (%): ',num2str(round(classificationRate*100))]); % display classification accuracy as well
