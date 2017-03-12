rootFolder = fullfile('/usr/local/MATLAB/R2016b/bin','prot');

categories = {'One_Ricon_Hill', 'Amo_Motel','Astar_Resort','Union_Shopping_Complex','Novogradac_Company','City_Office', 'Furniture_Store','Houseno23','Saint_Apartments','Saint_Church','SF_Arcade', 'SF_Resort','Thompsons_sons_inc','Houseno24','Union_apartments'};
%% 
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');


countEachLabel(imds)
[trainingSet, val_testset] = splitEachLabel(imds, 0.80, 'randomize');
[validationset, testset]=splitEachLabel(val_testset, 0.5, 'randomize');
% Find the first instance of an image for each category
One_Ricon_Hill = find(trainingSet.Labels == 'One_Ricon_Hill', 1);
Amo_Motel = find(trainingSet.Labels == 'Amo_Motel', 1);
Astar_Resort = find(trainingSet.Labels == 'Astar_Resort', 1);
Union_Shopping_Complex = find(trainingSet.Labels == 'Union_Shopping_Complex', 1);
Novogradac_Company = find(trainingSet.Labels == 'Novogradac_Company', 1);
City_Office = find(trainingSet.Labels == 'City_Office', 1);
Furniture_Store = find(trainingSet.Labels == 'Furniture_Store', 1);
Houseno23 = find(trainingSet.Labels == 'Houseno23', 1);
Saint_Apartments = find(trainingSet.Labels == 'Saint_Apartments', 1);
Saint_Church = find(trainingSet.Labels == 'Saint_Church', 1);
SF_Arcade = find(trainingSet.Labels == 'SF_Arcade', 1);
SF_Resort = find(trainingSet.Labels == 'SF_Resort', 1);
Thompsons_sons_inc = find(trainingSet.Labels == 'Thompsons_sons_inc', 1);
Houseno24 = find(trainingSet.Labels == 'Houseno242', 1);
Union_apartments = find(trainingSet.Labels == 'Union_apartments', 1);

% figure
subplot(1,5,1);
imshow(readimage(trainingSet,One_Ricon_Hill))
subplot(1,5,2);
imshow(readimage(trainingSet,Amo_Motel))
subplot(1,5,3);
imshow(readimage(trainingSet,Astar_Resort))
subplot(1,5,4);
imshow(readimage(trainingSet,Union_Shopping_Complex))
subplot(1,5,5);
imshow(readimage(trainingSet,Novogradac_Company))




bag = bagOfFeatures(trainingSet);
img = readimage(imds, 1);
featureVector = encode(bag, img);

% Plot the histogram of visual word occurrences
figure
bar(featureVector)
title('Visual word occurrences')
xlabel('Visual word index')
ylabel('Frequency of occurrence')
categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);
confMatrix = evaluate(categoryClassifier, trainingSet);
confMatrix = evaluate(categoryClassifier, validationset);

% Compute average accuracy
mean(diag(confMatrix));

AM1 = find(testset.Labels == 'SF_Resort', 1);
% figure subplot(1,1,1);
test1=readimage(testset,AM1);
img1=test1;
[labelIdx, scores] = predict(categoryClassifier, img1);
categoryClassifier.Labels(labelIdx)
I = img1;
   position = [23 373];
box_color = {'red'};

RGB = insertText(I,position,categoryClassifier.Labels(labelIdx),'FontSize',18,'BoxColor',...
    box_color,'BoxOpacity',0.4,'TextColor','white');
figure
imshow(RGB)
title('Board');


RC = find(testset.Labels == 'City_Office', 1);
% figure subplot(1,1,1);
test3=readimage(testset,RC);
img3=test3;
[labelIdx, scores] = predict(categoryClassifier, img3);
categoryClassifier.Labels(labelIdx)
I3 = img3;
   position = [23 373];
box_color = {'red'};

RGB = insertText(I3,position,categoryClassifier.Labels(labelIdx),'FontSize',18,'BoxColor',...
    box_color,'BoxOpacity',0.4,'TextColor','white');
figure
imshow(RGB)
title('Board');

FS1 = find(testset.Labels == 'Furniture_Store', 1);
% figure
% subplot(1,1,1);
test2=readimage(testset,FS1);

img2 =test2;
[labelIdy, scores] = predict(categoryClassifier, img2);

% Display the string label
categoryClassifier.Labels(labelIdy)

I2 = img2;

   position = [23 373];
box_color = {'red'};

RGB = insertText(I2,position,categoryClassifier.Labels(labelIdy),'FontSize',18,'BoxColor',...
    box_color,'BoxOpacity',0.4,'TextColor','white');
figure
imshow(RGB)
title('Board');AM1 = find(testset.Labels == 'SF_Resort', 1);
% figure subplot(1,1,1);
test1=readimage(testset,AM1);
img1=test1;
[labelIdx, scores] = predict(categoryClassifier, img1);
categoryClassifier.Labels(labelIdx)
I = img1;
   position = [23 373];
box_color = {'red'};

RGB = insertText(I,position,categoryClassifier.Labels(labelIdx),'FontSize',18,'BoxColor',...
    box_color,'BoxOpacity',0.4,'TextColor','white');
figure
imshow(RGB)
title('Board');


RC = find(testset.Labels == 'City_Office', 1);
% figure subplot(1,1,1);
test3=readimage(testset,RC);
img3=test3;
[labelIdx, scores] = predict(categoryClassifier, img3);
categoryClassifier.Labels(labelIdx)
I3 = img3;
   position = [23 373];
box_color = {'red'};

RGB = insertText(I3,position,categoryClassifier.Labels(labelIdx),'FontSize',18,'BoxColor',...
    box_color,'BoxOpacity',0.4,'TextColor','white');
figure
imshow(RGB)
title('Board');

FS1 = find(testset.Labels == 'Furniture_Store', 1);
% figure
% subplot(1,1,1);
test2=readimage(testset,FS1);

img2 =test2;
[labelIdy, scores] = predict(categoryClassifier, img2);

% Display the string label
categoryClassifier.Labels(labelIdy)

I2 = img2;

   position = [23 373];
box_color = {'red'};

RGB = insertText(I2,position,categoryClassifier.Labels(labelIdy),'FontSize',18,'BoxColor',...
    box_color,'BoxOpacity',0.4,'TextColor','white');
figure
imshow(RGB)
title('Board');