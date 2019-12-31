import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder


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

# prepare input data
def prepare_inputs(X_train,X_test):
    oe=OrdinalEncoder()
    oe.fit(X_train)
    X_train_enc=oe.transform(X_train)
    X_test_enc=oe.transform(X_test)
    return X_train_enc,X_test_enc


# Prepare target
def prepare_targets(y_train,y_test):
    le=LabelEncoder()
    le.fit(y_train)
    y_train_enc=le.transform(y_train)
    y_test_enc=le.transform(y_test)
    return y_train_enc,y_test_enc

# load the dataset
X, y = load_dataset('breast-cancer.csv')
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# prepare input data
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
# prepare output data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)