%This function creates the oversampled dataset

function [x_oversampled,y_oversampled]=oversample(x_train,y_train)

vector_negative_indexes=find(y_train(:,end)==0);

vector_positive_indexes=find(y_train(:,end)==1);

number_of_desired=length(vector_negative_indexes)-length(vector_positive_indexes);

if number_of_desired>length(vector_positive_indexes)
    new_indexes_to_add=randsample(length(vector_positive_indexes),number_of_desired,true);
else
    new_indexes_to_add=randsample(length(vector_positive_indexes),number_of_desired,false);
end

add_these=vector_positive_indexes(new_indexes_to_add);

y_oversampled=vertcat(y_train,y_train(add_these,:));

x_oversampled=vertcat(x_train,x_train(add_these,:));