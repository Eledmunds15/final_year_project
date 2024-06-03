%% Image Processing: Defect Detection and Dataset Development
% Author: Ethan Edmunds

% Create array of indexes to iterate through all micrographs available
image_num = [1 2 3 4 5 6 7 8 9 10 11 12 13 14];

% Iterate through the array of indexes and apply the processImage function
for i = 1:length(image_num)
    processImage(i)
end

% A function that takes the index of an image and processes it
function m = processImage(image_no)
    
    % Identify the image that needs to be processed
    image_name = "image_"; % identify the image name
    image_format = ".tif"; % identify the format of the image
    directory = "IN713C_original_micrographs\\"; % identify the parent directory which stores all of the images
    full_name = directory + image_name + image_no + image_format; % Create full name for image file to be downloaded
    
    % Download and show the image
    img = imread(full_name);
    figure;
    imshow(img);

    img_gs = rgb2gray(img); % Grayscale the image
    img_gs = img_gs < 70; % Using thresholding to identify pixels with a brightness less than that of the threshold

    [rows, cols] = size(img_gs); % Find the size of the image

    img_gs(round(rows*0.95):end , : ) = []; % Crop out the bottom 5% of the image to remove the scale

    img_gs = imclearborder(img_gs); % Get rid of all images on the border
    img_gs = bwareafilt(img_gs, [20 inf]); % Filter out small defects with an area of less than 20

    def_ds = regionprops(img_gs, "Image", "BoundingBox", "Area"); % Creates a datastore containing information and an image of each defect
    
    % Show the thresholded image after processing (removal of bordering
    % defects and small defects
    figure;
    imshow(img_gs);
    hold on

    % Draw a box around each defect in the filtered image
    for i = 1:length(def_ds)
        rectangle(Position=def_ds(i).BoundingBox, EdgeColor='r', LineWidth = 1)
    end
    hold off

    % Sort the datastore based on area
    def_ds_table = struct2table(def_ds)
    def_ds_table = sortrows(def_ds_table, "Area", "ascend");
    def_ds = table2struct(def_ds_table);

    fprintf("The number of defects in the image after filtering is: " + length(def_ds) + "\n\n");

    % Iterate through each defect in the image and download them
    figure
    for i = 1:length(def_ds)
        
        % Show each individual defect in the micrograph
        target_def = def_ds(i).Image;
        subplot(1,1,1), imshow(target_def, "Border", "loose")

        % Create the defect image name
        defectImage_name = "unsorted_images\\" + image_name + image_no + "_" + string(i) + ".tif";

        % Write the image to a file
        imwrite(target_def, defectImage_name);

    end
    
    m = 0;
    fprintf("Image " + image_no + " has been processed...\n\n")

end