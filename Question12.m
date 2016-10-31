load('data/digit-dataset/test.mat')
load('data/digit-dataset/train.mat')
addpath liblinear-2.01/matlab

training_matrix = reshape(train_images,[784, 60000])';
temp_variable = training_matrix;
[testing_instance_matrix, count] = datasample(temp_variable,10000);
testing_vector = train_labels(count);
number_train = [100;200;500;1000;2000;5000;10000];
errors = [0 0 0 0 0 0 0];
confusion_matrix = ones(10,10,7);
accuracies = [1 1 1 1 1 1 1 ];

for j = 1:7
    [training_instance_matrix, tr_count] = datasample(temp_variable, number_train(j));
    training_label = train_labels(tr_count);
    model = train(training_label,sparse(training_instance_matrix));
    [predicted_label, accuracy, decision_values] = predict(testing_vector, sparse(testing_instance_matrix), model );
    accuracies(j) =accuracy(1);
    confusion = confusionmat(testing_vector,predicted_label);
    confusion_matrix(:,:,j) = confusion;
    
end
%Answer to Question 1
plot(number_train, accuracies,'o-')

%Answer to Question 2
confusion_plot(confusion_matrix) 



            
            
    





            
    
    
    





