%% Question 7
%  NOTE: I used a function called logmvnpdf from an external library 
%      that I am including in my code


load('/Users/bsatheesh/Desktop/hw4/data/train.mat')
load('/Users/bsatheesh/Desktop/hw4/data/test.mat')
load('/Users/bsatheesh/Desktop/hw4/data/spam_data.mat')
%%
%Intial ground work
lvl = graythresh(train_images);
train_images = im2bw(train_images, lvl);
training_matrix = reshape(train_images,[784, 60000])';
%training_matrix = zscore(training_matrix);
norm_train = training_matrix/norm(training_matrix);
one = ones(60000,1)';
%%
%Part (a) 
stoned = one*norm_train;
me = ones(1,784);
st = ones(1,784);
prior = ones(10,1);
for i = 1:10
g = norm_train(train_labels==i-1,:);
me(i,:) = mean(g);
st(i,:) = std(g);
temp = cov(g);
covar(i,:,:) = temp;
prior(i) = length(g)/60000; %Part (b) of the solution
%figure(i)
%plot(cov_mat) %To visualise the Covariance matrices Part (C)
end
%figure(11)
%plot(prior) % Plots the prior probabilities 
%%
% Find the loss Part (D)
cov_mean = mean(covar);
cov_mean = reshape(cov_mean,[784,784]);
combined_mat = [train_labels training_matrix];
validation_set = datasample(combined_mat,10000);
val_label = validation_set(:,1);
val_matrix = validation_set(:,2:785);
y_validation = ones(10000,1);
tots = ones(10,1);
error = ones(9,1);
temp = zeros(784);
num_samples = [100 200 500 1000 2000 5000 10000 30000 50000];
for index = 1:length(num_samples)
    train_set = datasample(combined_mat,num_samples(index));
    train_label = train_set(:,1);
    train_matrix = train_set(:,2:785); 
        
    for sub_calc = 1:10
        g = train_matrix(train_label==i-1,:);
        temp = temp + cov(g);
    end
    cov_mean = temp/10;
    
    for i = 1:10
        g = train_matrix(train_label==i-1,:);
        me = mean(g);
        covaraince_matrix = cov(g);
        %covar(i,:,:) = temp;  
        prior = length(g)/60000; %Part (b) of the solution
        tots = log(prior)*ones(1,10000)+logmvnpdf(val_matrix,me,(covaraince_matrix+0.790*eye(784)));
        posterior(:,i) = tots';
    end
    [maximum ,y_validation] = max(posterior');
    er = (y_validation-1)'~=val_label;
    error(index) = sum(double(er))/100;
    end
figure(1)
plot(error)
hold on
legend('Diiferent mu and sigma')
xlabel('Percentage Error')
ylabel('Error Rate')
hold off
%%
% Kaggle submission using the optimum classification
a = reshape(test_images,[10000,28,28]);
test_im = reshape(a,[10000,784]);


val_label = double(test_labels);
val_matrix = double(test_im);
tots = ones(10,1);
temp = zeros(784);
num_samples = 60000
    train_set = datasample(combined_mat,num_samples);
    train_label = train_set(:,1);
    train_matrix = train_set(:,2:785); 
        
    for sub_calc = 1:10
        g = train_matrix(train_label==i-1,:);
        temp = temp + cov(g);
    end
    cov_mean = temp/10;
    
    for i = 1:10
        g = train_matrix(train_label==i-1,:);
        me = mean(g);
        covaraince_matrix = cov(g);
        %covar(i,:,:) = temp;  
        prior = length(g)/60000; %Part (b) of the solution
        tots = log(prior)*ones(1,10000)+logmvnpdf(val_matrix,me,(covaraince_matrix+0.790*eye(784)));
        posterior(:,i) = tots';
    end
    [maximum ,y_validation] = max(posterior');
    er = (y_validation-1)'~=val_label;
    error = sum(double(er))/100;
    
%%
%Spammy Data
spam_test = test_data;
spam_test = spam_test/norm(spam_test);
sp_train_image = training_data;
sp_train_image = sp_train_image/norm(sp_train_image);
sp_train_label = double((training_labels)');
posterior = ones(5857,2);

val_label = ones(5857,1);
val_matrix = double(spam_test);
tots = ones(10,1);
num_samples = 5172;
    
    train_label = sp_train_label;
    train_matrix = sp_train_image; 
    
    for i = 1:2
        g = train_matrix(train_label==i-1,:);
        me = mean(g);
        covaraince_matrix = cov(g);
        %covar(i,:,:) = temp;  
        prior = length(g)/60000; %Part (b) of the solution
        tots = (log(prior)*ones(1,5857))+logmvnpdf(val_matrix,me,(covaraince_matrix+2.5*eye(32)));
        posterior(:,i) = tots';
    end
    [maximum ,y_val] = max(posterior');
    val_label = (y_val-1)';












