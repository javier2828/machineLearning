% Javier Salazar 1001144647
% HW5 EM Algorithm on GMM
clc
%----input arguments---------
trainName = 'Training.txt'; % training and test files
testName = 'Test-1.txt';
gaussians = 5; % how many mixtures we want
classes = 10; % total number of classes
%-------import data------------
trainData = readmatrix(trainName);
testData = readmatrix(testName);
features = size(trainData,2)-1;
testError = zeros(1,gaussians);
%-------get results for different number of mixtures
for Q = 1:gaussians
    %---initial values for gaussian statistical parameters---
    class = cell(classes,1); % store points for specific classes
    m = cell(classes,1); M = cell(classes,1); D = cell(classes,1); % for mean calculation
    means = zeros(Q,features,classes); dev = zeros(Q,features,classes); weights = zeros(Q,classes); % the 3 parameters
    priors = zeros(1,classes); % frequentist approach for prior probabilities
    probMatrix = cell(classes,1); % matrix for optimizing parameters
    for i = 1:classes % go through classes
        class{i} = trainData(trainData(:,features+1)==i,1:features); % add data points
        m{i} = min(class{i}, [], 1);
        M{i} = max(class{i}, [], 1);
        D{i} = (M{i}-m{i})/Q; % these 3 needed for mean
        priors(i) = size(class{i}, 1)/size(trainData,1); % how many data points do we have per class
        for k = 1:Q % go through for all mixtures
            means(k,:,i) = m{i} + 0.5*D{i} + (k-1)*D{i}; % save mean for specific class and specific mixture
            dev(k,:,i) = 0.1.*ones(1,features); % initial deviations as specified
            weights(k,i) = (1/Q); % how much weight per mixture
        end
    end
    %------main EM iteravitve method------------
    log_like = 1000; old_like = 1;
    for c = 1:classes % for each class until optimized
        while (abs(log_like - old_like) > 1e-8)
            old_like = log_like;
            log_like = 0;
            probMatrix{c} = []; % needed since I append the cell array later on
            %--------expectation---------
            for i = 1:Q % get probability matrix for all gaussians
                Num = weights(i,c).*multipdf( class{c}, means(i,:,c), dev(i,:,c)); % take weight into accounts
                probMatrix{c} = [probMatrix{c}; transpose(Num)]; % append to the matrix
                log_like = log_like + sum(log2(Num)); % keep track of cost function as we go along
            end
            prob = probMatrix{c};
            Den = repmat(sum(prob, 1),Q,1); % scale probabilites by total
            probMatrix{c} = prob./Den; % matrix used during two step optimization method
            %-------maximazation------------
            probSum = sum(prob,2); % scale prob values per gaussian
            for i = 1:Q
                % calculate mean by multyplying each data point row by prob
                % scalar and sum result
                means(i,:,c) = sum(transpose(prob(i,:)).*class{c}, 1)./(probSum(i));
                temp = (class{c}-means(i,:,c)).^2;
                % similar for sigmas (deviations)
                sigma = sqrt( sum(transpose(prob(i,:)).*temp)./(probSum(i)) );
                % this line is for numerical stability
                sigma( sigma < 0.01) = 0.01;
                dev(i,:,c) = sigma;
                % weight scaled approriatly
                weights(i,c) = (probSum(i))/sum(probSum);
            end
        end
        log_like = 1000;
    end
    %---------testing error---------------
    trueValues = testData(:,features+1); % labels
    predMatrix = zeros(size(trueValues,1),classes); % get values to see maximum for each point
    for c = 1:classes % go through all classes
        total = 0;
        for i = 1:Q % create total value from all gaussians and respective weights
            total = total+weights(i,c).*multipdf(testData(:,1:features), means(i,:,c), dev(i,:,c));
        end
        predMatrix(:,c) = priors(c).*total; % multiply by prior to get unscaled posterior values
    end
    [~, predValues] = max(predMatrix, [], 2); % find maximum in each row
    % calculate how many classes are wrong
    testError(Q) = (length(predValues)-sum(eq(trueValues,predValues)))/length(predValues);
    disp(['The testing classification error probability for ',num2str(Q), ' gaussians per class is: ', num2str(testError(Q)) ]);
    %-----confusion matrix generation-----------
    confusion = zeros(10,10);
    for i = 1:size(predValues,1)
        % if class 4 is misclassified as class 3 then p43 count increases
        confusion(predValues(i), trueValues(i)) = confusion(predValues(i), trueValues(i))+1;
    end
    % scale values to get probability and not counts
    scalar = sum(confusion,2); scalar = 1./scalar; scalar(scalar==Inf)=0;
    confusion = transpose(confusion.*scalar);
end
%---------generate plot----------------------------
figure
scatter(1:gaussians,testError, 500, 'x', 'LineWidth', 3);
title('Expectation-Maximization on Gaussian Mixture Models (Classification)', 'FontSize', 20);
xlabel('Number of Gaussian Mixtures', 'FontSize', 15);
ylabel('Testing Classification Error', 'FontSize', 15)
%---------generate plot for confusion matrix---------
%------VISUALIZATION PLOT TAKEN FROM HERE: 
% https://stackoverflow.com/questions/3942892/how-do-i-visualize-a-matrix-with-colors-and-values-displayed
figure
mat = confusion;
imagesc(mat);            % Create a colored plot of the matrix values
colormap(flipud(gray));  % Change the colormap to gray (so higher values are
                         %   black and lower values are white)

textStrings = num2str(mat(:), '%0.3f');       % Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  % Remove any space padding
[x, y] = meshgrid(1:classes);  % Create x and y coordinates for the strings
hStrings = text(x(:), y(:), textStrings(:), ...  % Plot the strings
                'HorizontalAlignment', 'center');
midValue = mean(get(gca, 'CLim'));  % Get the middle value of the color range
textColors = repmat(mat(:) > midValue, 1, 3);  % Choose white or black for the
                                               %   text color of the strings so
                                               %   they can be easily seen over
                                               %   the background color
set(hStrings, {'Color'}, num2cell(textColors, 2));  % Change the text colors
title('Confusion Matrix (5 gaussians per class)','FontSize',20);
xlabel('Class', 'FontSize', 15);
ylabel('Class (columns add to 1)', 'FontSize', 15);

set(gca, 'XTick', 1:classes, ...                             % Change the axes tick marks
         'XTickLabel', {'1', '2', '3', '4', '5','6', '7', '8', '9', '10' }, ...  %   and tick labels
         'YTick', 1:classes, ...
         'YTickLabel', {'1', '2', '3', '4', '5','6', '7', '8', '9', '10'}, ...
         'TickLength', [0 0]);
%---------pdf function------------
function values = multipdf(X, mean, sigma)
var = length(mean);
values = ones(size(X,1),1);
for i = 1:var
    values = values.*normpdf(X(:,i),mean(i), sigma(i));
end
end
