function [predictions] = createPredictions(mdl, validation_images, classNames)
    
    % create predictions using the model and the validation images
    [predictions, ~] = predict(mdl, validation_images);

    % use one hot decode to conver the predictions back into a categorical
    % array and find it's transpose
    predictions = transpose(onehotdecode(predictions, classNames, 1));

end