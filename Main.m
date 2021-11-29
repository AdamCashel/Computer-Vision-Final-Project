clear;
%Gettting training_faces into matrix
Location = 'C:\Users\adamc\OneDrive\Desktop\Final Computer Vision Project\Computer-Vision-Final-Project\training_test_data\training_faces\*bmp';
current_face_size = 100;
current_nonface_size = 100;
ds = imageDatastore(Location);
length = 5047; %Number of pictures in training_faces
faces = zeros(50,50,100); %Matrix to hold current training face pics
total_facepics = zeros(50,50,length); % Matrix to hold all training face pics 
total_nonfacepics = zeros(50,50,length); % Matrix to hold all training nonface pics 
counter = 1;

while hasdata(ds)
    tempImage = read(ds);
    tempImage = mat2gray(tempImage);
    tempImage = tempImage(1:50,1:50);
    total_facepics(:,:,counter) = tempImage;
    %faces(:,:,counter) = tempImage;
    
    counter = counter + 1;
end

%Gettting training_nonfaces into matrix
Location2 = 'C:\Users\adamc\OneDrive\Desktop\Final Computer Vision Project\Computer-Vision-Final-Project\training_test_data\training_nonfaces\*jpg';
ds2 = imageDatastore(Location2);
length = 5047; %Number of pictures in training_faces
nonfaces = zeros(50,50,100);
counter = 1;

while hasdata(ds2)
    tempImage = read(ds2);
    tempImage = mat2gray(tempImage);
    tempImage = tempImage(1:50,1:50);
    total_nonfacepics(:,:,counter) = tempImage;
    %nonfaces(:,:,counter) = tempImage;
    
    counter = counter + 1;
end

%Get only the first 100 training pics of face and nonface
faces(:,:,1:100) = total_facepics(:,:,1:100);
nonfaces(:,:,1:100) = total_nonfacepics(:,:,1:100);


% choosing a set of random weak classifiers
number = 10000;
face_vertical = 50;
face_horizontal = 50;
weak_classifiers = cell(1, number);
for i = 1:number
    weak_classifiers{i} = generate_classifier(face_vertical, face_horizontal);
end

% save classifiers1000 weak_classifiers

%Get face_integrals
face_integrals = zeros(50,50,current_face_size);

for i = 1:current_face_size
    face_integrals(:,:,i) = integral_image(faces(:,:,i));
    
end

%Get nonface_integrals
nonface_integrals = zeros(50,50,current_nonface_size);

for i = 1:current_nonface_size
    nonface_integrals(:,:,i) = integral_image(nonfaces(:,:,i));
     
end




face_vertical = 50;
face_horizontal = 50;




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

%Calling adaboost to find error rate of strong classifier with 50 rounds of
%training
boosted_classifier = AdaBoost(responses, labels, 50)



% Let's classify a couple of our face and non-face training examples. 
% A positive prediction value means the classifier predicts the input image to be
% a face. A negative prediction value means the classifier thinks it's not
% a face. Values farther away from zero means the classifier is more
% confident about its prediction, either positive or negative.

prediction = boosted_predict(total_facepics(:, :, 200), boosted_classifier, weak_classifiers, 50);

prediction = boosted_predict(total_nonfacepics(:, :, 500), boosted_classifier, weak_classifiers, 50);

%Bootstraping Section

%Detecting nonfaces
numberpics = 0;
wrong = 0;
for u = 1:8
for j = (u*100)+1:(u*100)+100
    prediction = boosted_predict(total_nonfacepics(:,:,j), boosted_classifier, weak_classifiers, 50);
    numberpics = numberpics + 1;
    %if prediction is less than 0 add to training data
    if prediction > 0
        current_nonface_size = current_nonface_size + 1;
        temp_matrix = zeros(50,50,current_nonface_size);
        temp_matrix = nonfaces(:,:,1:current_nonface_size-1);
        temp_matrix(:,:,current_nonface_size) = total_nonfacepics(:,:,j);
        nonfaces = zeros(50,50,current_nonface_size);
        nonfaces(:,:,1:current_nonface_size) = temp_matrix(:,:,1:current_nonface_size);
        
        %Get and Add integral nonface picture
        tempmatrix = zeros(50,50,current_nonface_size);
        tempmatrix(:,:,1:current_nonface_size-1) = nonface_integrals(:,:,1:current_nonface_size-1);
        tempmatrix(:,:,current_nonface_size) = integral_image(total_nonfacepics(:,:,j));
        nonface_integrals = zeros(50,50,current_nonface_size);
        nonface_integrals(:,:,1:current_nonface_size) = tempmatrix(:,:,1:current_nonface_size);
        
        wrong = wrong + 1;
    end
end
end

%Dectecting faces
%Add threshold other than 1
%Number of rounds

for r = 1:8
for k = (r*100)+1:(r*100)+100
     prediction = boosted_predict(total_facepics(:,:,k), boosted_classifier, weak_classifiers, 50);
     numberpics = numberpics + 1;
    %if prediction is less than 0 add to training data
    if prediction < 0
        current_face_size = current_face_size + 1;
        temp_matrix = zeros(50,50,current_face_size);
        temp_matrix = faces(:,:,1:current_face_size-1);
        temp_matrix(:,:,current_face_size) = total_facepics(:,:,k);
        faces = zeros(50,50,current_face_size);
        faces(:,:,1:current_face_size) = temp_matrix(:,:,1:current_face_size);
        
        %Get and Add integral face picture
        tempmatrix = zeros(50,50,current_face_size);
        tempmatrix(:,:,1:current_face_size-1) = face_integrals(:,:,1:current_face_size-1);
        tempmatrix(:,:,current_face_size) = integral_image(total_facepics(:,:,k));
        face_integrals = zeros(50,50,current_face_size);
        face_integrals(:,:,1:current_face_size) = tempmatrix(:,:,1:current_face_size);
        
        wrong = wrong + 1;
    end
end
end


wrong
correct = numberpics - wrong




photo = read_gray('faces4.bmp');

% rotate the photograph to make faces more upright (we 
% are cheating a bit, to save time compared to searching
% over multiple rotations).
photo2 = imrotate(photo, -10, 'bilinear');
photo2 = imresize(photo2, 0.34, 'bilinear');
figure(1); imshow(photo2, []);
tic; result = boosted_multiscale_search(photo2, 1, boosted_classifier, weak_classifiers, [50, 50]); toc
tic; [result, boxes] = boosted_detector_demo(photo2, 1, boosted_classifier, weak_classifiers, [50, 50], 2); toc
figure(2); imshow(result, []);
figure(3); imshow(max((result > 4) * 255, photo2 * 0.5), [])



%Classifier Cascades
%Result is either a face or not a face

for f = 1:current_face_size
    %result = cascade_classify();
end








