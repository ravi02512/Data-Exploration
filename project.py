import pandas as pd
import numpy as np
from sklearn import metrics

data=pd.read_csv('train.csv')
print(np.shape(data))
cols=data.columns[data.isna().any()]  ##columns with missing values and  we have to predict

class Handling_missing_data(): 
    def __init__(self, train_data, test_data, predict_column):
        self.train_data=train_data
        self.test_data=test_data
        self.predict_column= predict_column

    def train_and_test_data(self):
        data= pd.read_csv(self.train_data)
        null_data=data[data.isnull().any(axis=1)]
        data=data.dropna(axis=0)
        X_train=data[data._get_numeric_data().columns].values.astype(float)
        Y_train=data[self.predict_column].values

        Test_data=pd.read_csv(self.test_data)
        Test_data=Test_data.dropna(axis=0)
        X_test=Test_data[data._get_numeric_data().columns].values.astype(float)
        Y_test=Test_data[self.predict_column].values
        return X_train,X_test,Y_train,Y_test

    def feature_scaling(self,x_train,x_test): ## optional(we can do feature scaling if we want) 
        self.x_train=x_train
        self.x_test=x_test

        from sklearn.preprocessing import StandardScaler
        sc=StandardScaler()
        x_train=sc.fit_transform(x_train)
        x_test=sc.transform(x_test)
        return x_train,x_test

    def Knn_model(self,x_train,y_train):

        from sklearn.neighbors import KNeighborsClassifier
        classifier=KNeighborsClassifier(n_neighbors=10, metric= 'minkowski', p=2)
        pred=classifier.fit(x_train,y_train)

        return pred    ## returns trained classifier

    def missing_data(self,train_data): ##this will create a dataframe with nan values and retrieve the dataframe with non nan values columns
        data=pd.read_csv(train_data)
        null_data=data[data.isnull().any(axis=1)] ##data with null values
        X_test=null_data[null_data._get_numeric_data().columns].values.astype(float)
        return X_test,null_data

#seperating dependent and independent variable from training and testing data
hmd=Handling_missing_data('train.csv','test.csv',cols[0])
x_train,x_test,y_train,y_test=hmd.train_and_test_data()

#extracting the dataframe of data with null values from the training data and preparing data for pridiction
x_test,null_data=hmd.missing_data('train.csv')

#fitting the training data in knn classifier
pred=hmd.Knn_model(x_train,y_train)

#predicting the missing data which is x_test
y_pred=pred.predict(x_test)

##creating the dataframe with predicted series for zero column
first=pd.DataFrame(y_pred, columns=[cols[0]])





##repeating the above set of procedure for all column values which have missing data
        
for column in cols[1:]:

    hmd=Handling_missing_data('train.csv','test.csv',column)

    x_train,x_test,y_train,y_test=hmd.train_and_test_data()

    x_test,null_data=hmd.missing_data('train.csv')
   
    y_pred=hmd.Knn_model(x_train,y_train)
    pred=y_pred.predict(x_test)
    second=pd.DataFrame(pred, columns=[column])
    third=pd.merge(first,second, left_index=True , right_index=True)
    first=third

first[None]=null_data.index
first=first.set_index(None)
new_data=null_data.drop(cols, axis=1)
first=pd.merge(first,new_data, left_index=True , right_index=True)
previous_data=data.dropna(axis=0)

final_data=pd.concat([previous_data,first], axis=0, ignore_index=True)
print(final_data)
    





