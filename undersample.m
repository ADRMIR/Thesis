%This function creates the oversampled dataset

function [x_undersampled,y_undersampled]=undersample(x_train,y_train)

vector_negative_indexes=find(y_train(:,end)==0);

vector_positive_indexes=find(y_train(:,end)==1);

number_of_desired=length(vector_negative_indexes)-length(vector_positive_indexes);

indexes_to_remove=randsample(length(vector_negative_indexes),number_of_desired,false);

y_train(indexes_to_remove,:)=[];

y_undersampled=y_train;

x_train(indexes_to_remove,:)=[];

x_undersampled=x_train;