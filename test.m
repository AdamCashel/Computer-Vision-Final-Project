%%Script test.m runs the final face detector with the test image data and
%%computes accuracy over that data
directory;
userpath(code_directory);
load ('boosted_classifier.mat');
load ('weak_classifiers.mat');

directory;
False_Positives = 0;
Total_Pics = 0;
False_Negatives = 0;
Correct = 0;
face_size = [50 50];

Location = 'test_face_photos\*jpg';
Location = append(data_directory,Location);
ds = imageDatastore(Location);
counter = 1;

while hasdata(ds)
    tempImage = read(ds);
    tempImage = rgb2gray(tempImage);
    Total_Pics = Total_Pics + 1;
    %[result, boxes] = boosted_detector_demo(photo2, 1, boosted_classifier, weak_classifiers, [50, 50], 4);
    %prediction = boosted_predict(tempImage, boosted_classifier, weak_classifiers, 50);
    %[max_responses, max_scales] = boosted_multiscale_search(tempImage, 1, boosted_classifier, weak_classifiers, face_size);
    result = apply_classifier_aux(tempImage, boosted_classifier, weak_classifiers, face_size);
                                       
    prediction = max(result(:));
    if prediction > .3
        Correct = Correct + 1;
    else
        False_Negatives = False_Negatives + 1;
    end
end


Location = 'test_nonfaces\*jpg';
Location = append(data_directory,Location);
ds = imageDatastore(Location);
counter = 1;

while hasdata(ds)
    tempImage = read(ds);
    tempImage = rgb2gray(tempImage);
    Total_Pics = Total_Pics + 1;
    %[result, boxes] = boosted_detector_demo(photo2, 1, boosted_classifier, weak_classifiers, [50, 50], 4);
    %prediction = boosted_predict(tempImage, boosted_classifier, weak_classifiers, 50);
    %[max_responses, max_scales] = boosted_multiscale_search(tempImage, 1, boosted_classifier, weak_classifiers, face_size);
    result = apply_classifier_aux(tempImage, boosted_classifier, weak_classifiers, face_size);
                                       
    prediction = max(result(:));
    if prediction >= 0
        Correct = Correct + 1;
    else
        False_Negatives = False_Negatives + 1;
    end
end



 
Location = 'test_cropped_faces\*bmp';
Location = append(data_directory,Location);
ds = imageDatastore(Location);
counter = 1;

while hasdata(ds)
    tempImage = read(ds);
    tempImage = mat2gray(tempImage);
    Total_Pics = Total_Pics + 1;
    %[result, boxes] = boosted_detector_demo(photo2, 1, boosted_classifier, weak_classifiers, [50, 50], 4);
    %prediction = boosted_predict(tempImage, boosted_classifier, weak_classifiers, 50);
    %[max_responses, max_scales] = boosted_multiscale_search(tempImage, 1, boosted_classifier, weak_classifiers, face_size);
    %result = SearchForFace(tempImage, boosted_classifier, weak_classifiers, face_size);
    result = apply_classifier_aux(tempImage, boosted_classifier, weak_classifiers, face_size);  
    
    prediction = max(result(:));
    if prediction >= 3
        Correct = Correct + 1;
    else
        False_Positives = False_Positives + 1;
    end
end


Total_Pics
Test_Correct_Percentage = (Correct / Total_Pics) * 100



