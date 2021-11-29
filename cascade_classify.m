function result = cascade_classify(window, boosted_classifier)

K = 60;

for i = 1:K-1
    
 
   prediction = boosted_predict(window, boosted_classifier, weak_classifiers, 50);
   %If prediction of window is less than .3 return not a face
   if prediction < .3
       result = 0;
       return;
   end
   

end

result = 1;
return;


end

