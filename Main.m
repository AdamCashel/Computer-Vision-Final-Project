clear;
%Gettting training_faces into matrix
Location = 'C:\Users\adamc\OneDrive\Desktop\Final Computer Vision Project\Computer-Vision-Final-Project\training_test_data\training_faces\*bmp';
current_face_size = 100;
current_nonface_size = 100;
ds = imageDatastore(Location);
length = 3047; %Number of pictures in training_faces
faces = zeros(100,100,100); %Matrix to hold current training face pics
total_facepics = zeros(100,100,length); % Matrix to hold all training face pics 
total_nonfacepics = zeros(100,100,length); % Matrix to hold all training nonface pics 
counter = 1;

while hasdata(ds)
    tempImage = read(ds);
    tempImage = mat2gray(tempImage);
    total_facepics(:,:,counter) = tempImage;
    %faces(:,:,counter) = tempImage;
    
    counter = counter + 1;
end

%Gettting training_nonfaces into matrix
Location2 = 'C:\Users\adamc\OneDrive\Desktop\Final Computer Vision Project\Computer-Vision-Final-Project\training_test_data\training_nonfaces\*jpg';
ds2 = imageDatastore(Location2);
length = 3047; %Number of pictures in training_faces
nonfaces = zeros(100,100,100);
counter = 1;

while hasdata(ds2)
    tempImage = read(ds2);
    tempImage = mat2gray(tempImage);
    tempImage = tempImage(1:100,1:100);
    total_nonfacepics(:,:,counter) = tempImage;
    %nonfaces(:,:,counter) = tempImage;
    
    counter = counter + 1;
end

%Get only the first 100 training pics of face and nonface
faces(:,:,1:100) = total_facepics(:,:,1:100);
nonfaces(:,:,1:100) = total_nonfacepics(:,:,1:100);


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
face_integrals = zeros(100,100,current_face_size);

for i = 1:current_face_size
    face_integrals(:,:,i) = integral_image(faces(:,:,i));
    
end

%Get nonface_integrals
nonface_integrals = zeros(100,100,current_nonface_size);

for i = 1:current_nonface_size
    nonface_integrals(:,:,i) = integral_image(nonfaces(:,:,i));
     
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

prediction = boosted_predict(total_facepics(:, :, 200), boosted_classifier, weak_classifiers, 15)

prediction = boosted_predict(total_nonfacepics(:, :, 500), boosted_classifier, weak_classifiers, 15)

%Bootstraping Section

%Detecting nonfaces
wrong = 0;
for j = 1:100
    prediction = boosted_predict(total_nonfacepics(:,:,j), boosted_classifier, weak_classifiers, 15)
    %if prediction is less than 0 add to training data
    if prediction > 0
        current_nonface_size = current_nonface_size + 1;
        temp_matrix = zeros(100,100,current_nonface_size);
        temp_matrix = nonfaces(:,:,1:current_nonface_size-1);
        temp_matrix(:,:,current_nonface_size) = total_nonfacepics(:,:,j);
        nonfaces = zeros(100,100,current_nonface_size);
        nonfaces(:,:,1:current_nonface_size) = temp_matrix(:,:,1:current_nonface_size);
        wrong = wrong + 1;
    end
end

%Dectecting faces
for k = 101:201
     prediction = boosted_predict(total_facepics(:,:,k), boosted_classifier, weak_classifiers, 15)
    %if prediction is less than 0 add to training data
    if prediction < 0
        current_face_size = current_face_size + 1;
        temp_matrix = zeros(100,100,current_face_size);
        temp_matrix = faces(:,:,1:current_face_size-1);
        temp_matrix(:,:,current_face_size) = total_facepics(:,:,k);
        faces = zeros(100,100,current_face_size);
        faces(:,:,1:current_face_size) = temp_matrix(:,:,1:current_face_size);
         wrong = wrong + 1;
    end
end

wrong
correct = 200 - wrong



