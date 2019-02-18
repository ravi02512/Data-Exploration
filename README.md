# predicting nan values using knn algorithm 
pridicting the nan values in a dataset with KNN algorithm

About the dataset:-
There is two employee dataset training and testing which contains nan values.

Algorithm used:-
KNN algorithm

Process:-
In this program we first extracting the dataframe which contain nan values so the rest of training data is without nan values.
split the data in dependent and indepedent values(select the column for x variable is numeric)
train the knn classifier
extract the x from the missing data frame put this x in knn classifier and you get the predicted column

loop the above process for columns with nan value
concatenate the predicted data frame with train dataframe without nan values 
fianlly you will get the dataframe with no nan values

