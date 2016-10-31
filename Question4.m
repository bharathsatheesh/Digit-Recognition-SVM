load('data/spam-dataset/spam_data.mat')
addpath liblinear-2.01/matlab

    % testing data
    training_labels = double(training_labels');
    [testing_instance_matrix, index] = datasample(training_data, 772);
    testing_label_vector = training_labels(index);
    %using 10 fold cross validation on rest
    train_labels_CV = training_labels;
    train_labels_CV(index) = [];
    
    train_images_CV = training_data;
    train_images_CV(index,:) = [];
  
    SampleMatrix = [train_images_CV train_labels_CV];
    SampleMatrix = SampleMatrix(randperm(size(SampleMatrix,1)),:);
    train_labels_CV = SampleMatrix(:,size(SampleMatrix,2));
    train_images_CV = SampleMatrix(:,1:size(SampleMatrix,2)-1);
    accuracies = [];
    validationAccuracies = [];
    CVal = [0.1:0.1:20];
    for j = 1:100
        c = CVal(j);
        for i = 1:10
            validationData = train_images_CV(440*(i-1)+1:440*i,:);
            validationLabel = train_labels_CV(440*(i-1)+1:440*i);
            trainingData = train_images_CV;
            trainingData(440*(i-1)+1:440*i,:) = [];
            trainLabel = train_labels_CV;
            trainLabel(440*(i-1)+1:440*i) = [];
            model = train(trainLabel, sparse(trainingData), cat(2,'-s 2 -c ',num2str(c)));
            [predicted_label, accuracy, decision_values] = predict(validationLabel, sparse(validationData), model);
            validationAccuracies(i) = accuracy(1);
        end
        accuracies(j) = sum(validationAccuracies)/10;    
    end
    best_c = CVal(find(accuracies == max(accuracies)));
    best_c = randsample(best_c, 1);
    model = train(train_labels_CV, sparse(train_images_CV), cat(2,'-s 2 -c ',num2str(best_c)));
    [predicted_label, accuracy, decision_values] = predict(testing_label_vector, sparse(test_data), model);
    out = [best_c accuracy(1)]
    


