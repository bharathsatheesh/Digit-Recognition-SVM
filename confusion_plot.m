function confusion_plot(confusion_matrix)
%This is my matrix to compute the color chart for the confusion matrix
size_matters = size(confusion_matrix);
for i = 1:size_matters(3)
    figure(i)
    imagesc(confusion_matrix(:,:,i))
    hold on;
end
hold off;

