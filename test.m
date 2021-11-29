%%Script test.m runs the final face detector with the test image data and
%%computes accuracy over that data

False_Positives = 0
Total_Pics = 0;
False_Negatives = 0;



%Get file names and test accuracy with boosted_classifier
file_names = nonface_filenames();

for i = 1:100
    
    file = file_names(i, 1);
    
end
