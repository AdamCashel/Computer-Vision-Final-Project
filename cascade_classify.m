function result = cascade_classify(window, boosted_classifier, weak_classifiers)
%Window is an image of a certain size
%Have to go through every 50x50 subwindow of the image to check for a face

K = 60;
result = 0;
i = 1;
Subwindow_Size = [50 50];

%Getting the strong classifier with a certian amount of weak classifiers
while (i < K && result == 0)
   number_weak_classifers = (4*i) - 1;
   
   %Go through all subwindows of window and call prediction_predict with that
   %subwindow and number_weak_classifers
   [image_rows, image_columns] = size(window);
   [template_rows, template_columns] = size(Subwindow_Size);
   row_start = floor(template_rows / 2) + 1;
   row_end = row_start + image_rows - 1;
   col_start = floor(template_columns / 2) + 1;
   col_end = col_start + image_columns - 2; %Changed from -1 was going out of bounds by 1
   %Get subwindow
   subwindow = window(row_start:row_end, col_start:col_end);
   
   % number_weak_classifers is the number of weak classifiers to use from
   % the weak_classifiers list
   prediction = boosted_predict(subwindow, boosted_classifier, weak_classifiers, number_weak_classifers);
   %Set the .3 to something that is dynamic where the first Ci gets rid of
   %most nonfaces
   if prediction < .3
       result = 0;
       return;
   end
   
end

result = 1;
return;

end

