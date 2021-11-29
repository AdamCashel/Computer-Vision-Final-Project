function result = cascade_classify(window, boosted_classifier)
%Window is an image of a certain size
%Have to go through every 50x50 subwindow of the image to check for a face

K = 60;
result = 0;
i = 1;
%Getting the strong classifier with a certian amount of weak classifiers
while (i < K && result == 0)
   number_weak_classifers = (4*i) - 1;
   
   %Go through a subwindow of window and call prediction_predict with that
   %subwindow and number_weak_classifers
   [image_rows, image_columns] = size( image);
   [template_rows, template_columns] = size(template);
   row_start = floor(template_rows / 2) + 1;
   row_end = row_start + image_rows - 1;
   col_start = floor(template_columns / 2) + 1;
   col_end = col_start + image_columns - 1;
   %Get subwindow
   subwindow = window(row_start:row_end, col_start:col_end);
   
   prediction = boosted_predict(subwindow, boosted_classifier, weak_classifiers, 50);
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

