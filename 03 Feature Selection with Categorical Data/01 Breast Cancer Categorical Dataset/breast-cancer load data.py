import pandas as pd
import numpy as np

# load dataset
# create function for load dataset
def load_dataset(filename):
    # load dataset as a pandas fataframe
    data=pd.read_csv(filename,header=None)
    
    # retrieve numpy  array
    dataset=data.values
    
    # Split into Input(X) and output(y) variables
    X=dataset[:,:-1]
    y=dataset[:,-1]
    
    # format all fileds as string
    X=X.astype(str)
    return X,y
  
# load the dataset
X,y=load_dataset("breast-cancer.csv")

# Split the data into train & test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=30,random_state=1)

# print the shape of train n test
print('Train : ', X_train.shape, y_train.shape)
print('Test  : ', X_test.shape,"", y_test.shape)