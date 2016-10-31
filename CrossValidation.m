function out = CrossValidation(input_mat,k)

length_input =  length(input_mat);
everything_vector = 1:length_input;
chosen_numbers = [];
rand_selected_matrix = ones(1000,1);
for i = 1:k
    counting_vector = setdiff(everything_vector,chosen_numbers);
    chosen_numbers = randperm(numel(counting_vector),floor(length_input/k));
    lel = input_mat(chosen_numbers,:);
    rand_selected_matrix = cat(2,rand_selected_matrix,chosen_numbers',lel);
    everything_vector = setdiff(everything_vector,chosen_numbers);
end
len = length(rand_selected_matrix);
rand_selected_matrix = rand_selected_matrix(:,2:len);
out = mat2cell(rand_selected_matrix,1000, 785*ones(10,1));
