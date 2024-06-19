%% Deep Learning: Convolutional Neural Networks
% Ethan Edmunds

clear; close all; clc

%% Loading Image Data

imageFolder = "defect_dataset"; % idetify name of folder with defect images
imds = imageDatastore(imageFolder, "IncludeSubfolders", true, "LabelSource", "foldernames"); % Create an image datastore for the dataset
tbl_count = countEachLabel(imds);

datasetSize = sum(tbl_count{:,"Count"});
fprintf("The number of defects in the dataset is " + string(datasetSize) + "\n");

%% Creating Training and Validation Datasets

[imdsTrain, imdsVal] = splitEachLabel(imds, 0.7, 0.3, "randomized");

tbl_count_training = countEachLabel(imdsTrain);
datasetSizeTrain = sum(tbl_count_training{:,"Count"});

tbl_count_validation = countEachLabel(imdsVal);
datasetSizeVal = sum(tbl_count_validation{:,"Count"});

%% Data augmentation

crack_preview = find(imdsTrain.Labels == 'Crack', 1); % find sample crack
pore_preview = find(imdsTrain.Labels == 'Pore', 1); % find sample pore
poreWithCrack_preview = find(imdsTrain.Labels == 'Pore with Crack', 1); % find sample pore with crack
lackOfFusion_preview = find(imdsTrain.Labels == 'Lack of Fusion', 1); % find sample lack of fusion

% Define the image augmentation configuration
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...    % Random horizontal flipping
    'RandYReflection', true, ...    % Random vertical flipping
    'RandRotation', [-20, 20], ...  % Random rotation between -10 and 10 degrees
    'RandScale', [0.9 1.1]);        % Random scaling

% 
imageSize = [200 200]; % define the target image size
augImdsTrain = augmentedImageDatastore(imageSize, imdsTrain, "DataAugmentation", imageAugmenter); % resize all of the images in the trianing dataset
augImdsVal = augmentedImageDatastore(imageSize, imdsVal, "DataAugmentation", imageAugmenter); % resize all of the images in the validation dataset

augDataVal = readall(augImdsVal); % turning the augmented image datastore (validation) into a table

% creating cell array of validation dataset to be used with dlarray objects
imagesVal = augDataVal{:, 1};
imagesVal = cat(4, imagesVal{:});

% get the validation labels
validationLabels = augDataVal{:,2};
classNames = categories(validationLabels); % get class names that are in the dataset

% turn the validation table of imgaes into a 
augImagesVal = dlarray(single(imagesVal), 'SSCB');

%% CNN: Baseline Model

num_classes = 4;
filter_size = 5;

num_layers = 4;
num_filters = 10;

aux_params = [num_classes, filter_size]; % create array of auxiliary parameters for network design
hyper_params_baseline = [num_layers, num_filters]; % create array of hyperparameters to be explored

baseline_options = trainingOptions('adam', ...
    'MiniBatchSize', 24, ...
    'InitialLearnRate',0.002, ...
    'MaxEpochs', 10, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augImdsVal, ...
    'ValidationFrequency',5, ...
    'Verbose',true, ...
    'ValidationPatience', 100, ...
    'Metrics', ["accuracy" "recall"], ...
    'OutputNetwork', 'best-validation', ...
    'Plots','training-progress');

%% create a baseline model
baseline_model_architecture = createNetworkModel(hyper_params_baseline, aux_params, imageSize);

% train baseline model
[baseMdl, baseMdl_info] = trainnet(augImdsTrain, baseline_model_architecture, "crossentropy", baseline_options);

%% evaluate this model
baseline_predictions = createPredictions(baseMdl, augImagesVal, classNames);

% plot the confusion matrix
baseline_evaluation = figure("Name", "Baseline Model Confusion Matrix");
plotconfusion(validationLabels, baseline_predictions);
title("Confusion Matrix: Baseline Model");

%% CNN: Grid Layer Search
% Create the dataset

% initialize a subset of the original data to train gridlayer search models
gs_train_ind = randsample(1:datasetSizeTrain, int32(datasetSizeTrain*0.25)); % Create a random number o
gs_subsetTrain = subset(imdsTrain, gs_train_ind);
datasubsetSizeTrain = countEachLabel(gs_subsetTrain);

gs_augsubsetTrain = augmentedImageDatastore(imageSize, gs_subsetTrain, "DataAugmentation", imageAugmenter);

%% Execute gridlayer search to find the optimal hyperparameters for nerual network design

gs_options = trainingOptions('adam', ...
    "MiniBatchSize", 40, ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs', 8, ...
    'Shuffle','every-epoch', ...
    'ValidationData', augImdsVal, ...
    'ValidationFrequency', 5, ...
    'ValidationPatience', 100, ...
    'Verbose',true ,...
    'Metrics', ["accuracy" "recall"], ...
    'OutputNetwork', 'best-validation');

gs_num_layers = [2, 3, 4, 5, 6, 7]; % number of layers for gridsearch to check
gs_num_filters = [5, 10, 15, 20, 25]; % number of filters for gridsearch to check

tic; % create a timer to measure runtime of gridlayer search

[gs_accuracy, gs_best_accuracy, gs_best_hyperparams] = gridlayerSearch(gs_augsubsetTrain, gs_num_layers, gs_num_filters, aux_params, imageSize, gs_options, augImagesVal, classNames, validationLabels);

gridlayer_runtime = toc; % record runtime of gridlayer search

%% show results in a 3D scatter plot
gs_accuracyTbl = array2table(gs_accuracy, "VariableNames", ["Network Blocks", "Network Filter Number", "Validation Accuracy"]);

figure("Name", "Gridsearch Accuracy Variation")
scatter3(gs_accuracyTbl, "Network Blocks", "Network Filter Number", "Validation Accuracy", "filled", "ColorVariable", "Validation Accuracy");
colorbar;

fprintf("Gridsearch optimal hyperparameters:\n");
fprintf("Number of Blocks: " + string(gs_best_hyperparams(1)) + " | Number of Filters: " + string(gs_best_hyperparams(2)) + "\n\n");

%% Train neural network using optimal hyper parameters
% create gridsearch optimised model
gsOptimised_model_architecture = createNetworkModel(gs_best_hyperparams, aux_params, imageSize);

% train gridsearch optimised model
[gsOptimisedMdl, gsOptimisedMdl_info] = trainnet(augImdsTrain, gsOptimised_model_architecture, 'crossentropy', baseline_options);

% evaluate this grid layer search optimised model
gsOptimised_predictions = createPredictions(gsOptimisedMdl, augImagesVal, classNames);

% plot the confusion matrix
gsOptimisedMdl_evaluation = figure("Name", "Gridsearch Optimised Model Confusion Matrix");
plotconfusion(validationLabels, gsOptimised_predictions);
title("Confusion Matrix: Gridsearch Optimised Model");

%% CNN: Bayesian Optimisation

bayesOpt_options = trainingOptions('adam', ...
    "MiniBatchSize", 25, ...
    'InitialLearnRate',0.002, ...
    'MaxEpochs', 10, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augImdsVal, ...
    'ValidationFrequency',5, ...
    'ValidationPatience', 100, ...
    'Verbose',true, ...
    'LearnRateDropPeriod', 1, ...
    'LearnRateDropFactor', 0.1, ...
    'Metrics', ["accuracy" "recall"], ...
    'OutputNetwork', 'best-validation');

% identify the variables to be optimised through bayesian optimization
optVars = [
    optimizableVariable('Layers', [2 10], "Type", "integer")
    optimizableVariable('Filters', [5 80], "Type", "integer")
    optimizableVariable('InitialLearnRate', [0.002 0.02], "Type", "real")
    optimizableVariable('MiniBatchSize', [24 128], "Type", "integer")
    optimizableVariable('LearnRateDropFactor', [0.1 0.5], "Type", "real")
    optimizableVariable('LearnRateDropPeriod', [1 5], "Type", "integer")
];

% use the makeObjFun function to create a objective function for bayesian
% optimization to evaluate
objFcn = makeObjFun(augImdsTrain, augImagesVal, validationLabels, classNames, bayesOpt_options, aux_params, imageSize);

tic; % start timer to measure total runtime

% execute bayesian optimization
BayesObject = bayesopt(objFcn, optVars, ...
    "MaxTime", 10*60*60, ...
    "IsObjectiveDeterministic", false, ...
    "UseParallel", false, ...
    "MaxObjectiveEvaluations", 100);

bayesOpt_runtime = toc; % record time taken for bayesian optimisation to take place

%% Find the file and load it
bayesFilesParentDir = "bayesian_optimisation_models_v2//";
bayesFilename = "8_26_0.084086";
bayesStruct = load(bayesFilesParentDir + bayesFilename);

% classify the validation output using the trained network
bayes_predictions = createPredictions(bayesStruct.trainedNet, augImagesVal, classNames);

% plot confusion matrix
figure;
plotconfusion(validationLabels,bayes_predictions)
title("Confusion Matrix: Bayesian Optimisation")

% accuracy in percent
bayes_accuracy = 100*sum(bayes_predictions == validationLabels)/numel(validationLabels);
disp("Basyesian optimization network accuracy is: " + num2str(bayes_accuracy) + "%");
%% Full Convolutional Network (Test)

fprintf("\n\n Script successfully run... \n\n")
