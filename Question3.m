load('data/digit-dataset/test.mat')
load('data/digit-dataset/train.mat')
addpath liblinear-2.01/matlab

training_matrix = reshape(train_images,[784, 60000])';

    
    
    %seperating testing data
    [testing_instance_matrix, index] = datasample(training_matrix, 60000);
    testing_label_vector = train_labels(index);
    %using 10 fold cross validation
    train_labels_vector = train_labels;
    train_labels_vector(index) = [];
    
    train_images_matrix = training_matrix;
    train_images_matrix(index,:) = [];
    
    
    [train_images_matrix, index] = datasample(train_images_matrix, 10000);
    train_labels_vector = train_labels_vector(index);
    
    shuffle_matrix = [train_images_matrix train_labels_vector];
    shuffle_matrix = shuffle_matrix(randperm(size(shuffle_matrix,1)),:);
    train_labels_vector = shuffle_matrix(:,size(shuffle_matrix,2));
    train_images_matrix = shuffle_matrix(:,1:size(shuffle_matrix,2)-1);
    accuracies = [];
    validationAccuracies = ones(1,10);
    CVal = [0.1:0.1:20];
    for j = 1:100
        c = CVal(j);
        for i = 1:10
            validationData = train_images_matrix(1000*(i-1)+1:1000*i,:);
            validationLabel = train_labels_vector(1000*(i-1)+1:1000*i);
            
            trainData = train_images_matrix;
            trainData(1000*(i-1)+1:1000*i,:) = [];
            trainLabel = train_labels_vector;
            trainLabel(1000*(i-1)+1:1000*i) = [];
            
            model = train(trainLabel, sparse(trainData), cat(2,'-s 2 -c ',num2str(c)));
            [predicted_label, accuracy, decision_values] = predict(validationLabel, sparse(validationData), model);
            validationAccuracies(i) = accuracy(1);
        end
        accuracies(j) = sum(validationAccuracies)/10;    
    end
    optimal_c = randsample(CVal(find(accuracies == max(accuracies))),1);
    model = train(train_labels_vector, sparse(train_images_matrix), cat(2,'-s 2 -c ',num2str(optimal_c)));
    [predicted_label, accuracy, decision_values] = predict((1:60000)', sparse(test_data), model);
    %Answer to Question 3
    
    output_C = optimal_c
    error_rate = 100 - max(accuracies)
    plot(CVal, accuracies);

