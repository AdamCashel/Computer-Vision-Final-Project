clear;
%Gettting training_faces into matrix
Location = 'C:\Users\adamc\OneDrive\Desktop\Final Computer Vision Project\Computer-Vision-Final-Project\training_test_data\training_faces\*bmp';
ds = imageDatastore(Location);
length = 3047; %Number of pictures in training_faces
faces = zeros(100,100,length);
counter = 1;
while hasdata(ds)
    tempImage = read(ds);
    faces(:,:,counter) = tempImage;
    
    counter = counter + 1;
end

%Gettting training_nonfaces into matrix
Location2 = 'C:\Users\adamc\OneDrive\Desktop\Final Computer Vision Project\Computer-Vision-Final-Project\training_test_data\training_nonfaces\*jpg';
ds2 = imageDatastore(Location2);
length = 3047; %Number of pictures in training_faces
nonfaces = zeros(100,100,length);
counter = 1;
while hasdata(ds2)
    tempImage = read(ds2);
    tempImage = tempImage(1:100,1:100);
    nonfaces(:,:,counter) = tempImage;
    
    counter = counter + 1;
end



% choosing a set of random weak classifiers
number = 1000;
face_vertical = 100;
face_horizontal = 100;
weak_classifiers = cell(1, number);
for i = 1:number
    weak_classifiers{i} = generate_classifier(face_vertical, face_horizontal);
end

% save classifiers1000 weak_classifiers

%Get face_integrals
face_integrals = zeros(100,100,3047);

for i = 1:3047
    %face_intergrals(:,:,i) = integral_image(faces(:,:,i));
    face_intergrals(:,:,i) = integralImage(faces(:,:,i));
end

%Get nonface_integrals
nonface_integrals = zeros(100,100,3047);

for i = 1:3047
    nonface_intergrals(:,:,i) = integral_image(faces(:,:,i));
     
end




face_vertical = 100;
face_horizontal = 100;




%%

%  precompute responses of all training examples on all weak classifiers

%clear all;
%load examples1000;
%load classifiers1000;

%load training faces and nonfaces

example_number = size(faces, 3) + size(nonfaces, 3);
labels = zeros(example_number, 1);
labels(1:size(faces, 3)) = 1;
labels((size(faces, 3)+1):example_number) = -1;
examples = zeros(face_vertical, face_horizontal, example_number);
examples (:, :, 1:size(faces, 3)) = face_integrals;
examples(:, :, (size(faces, 3)+1):example_number) = nonface_integrals;

classifier_number = numel(weak_classifiers);

responses =  zeros(classifier_number, example_number);

for example = 1:example_number
    integral = examples(:, :, example);
    for feature = 1:classifier_number
        classifier = weak_classifiers {feature};
        %eval_weak_classifier(classifier, integral);
        responses(feature, example) = eval_weak_classifier(classifier, integral);
    end
    disp(example);
end

% save training1000 responses labels classifier_number example_number

%Calling adaboost to find error rates of 25 strong classifiers
boosted_classifier = AdaBoost(responses, labels, 15);



% Let's classify a couple of our face and non-face training examples. 
% A positive prediction value means the classifier predicts the input image to be
% a face. A negative prediction value means the classifier thinks it's not
% a face. Values farther away from zero means the classifier is more
% confident about its prediction, either positive or negative.

prediction = boosted_predict(faces(:, :, 200), boosted_classifier, weak_classifiers, 15)

prediction = boosted_predict(nonfaces(:, :, 500), boosted_classifier, weak_classifiers, 15)


