function networkLayerArchitecture = createNetworkModel(hyper_params, aux_params, imageSize)
    
    num_layers = hyper_params(1); % identify number of layers to be used
    num_filters = hyper_params(2); % identify number of filters to be used

    num_classes = aux_params(1); % number of classes to be classified into
    filter_size = aux_params(2); % size of the filters used in network design

    % create input layer
    networkLayerArchitecture = [imageInputLayer([imageSize 1])];

    % create convolutional layers
    % this includes creating blocks of convolution -> batch normalisation
    % -> relu -> max pooling layers

    for i = 1:num_layers
        networkLayerArchitecture = [networkLayerArchitecture
            convolution2dLayer(filter_size, num_filters, "Padding", "same")
            batchNormalizationLayer
            reluLayer
            maxPooling2dLayer(2,"Stride",2,"Padding","same")
        ];
    end

    % output layers
    networkLayerArchitecture = [networkLayerArchitecture
        fullyConnectedLayer(num_classes)
        softmaxLayer
        % classificationLayer
    ];
end