import pandas as pd
from src.data.make_dataset import load_data
from src.models.train_model import split_data, Linear_Regression, Decision_TreeRegressor, RandomForest

#Data Path Definition
if __name__=="__main__":
    #Load Data
    data_path = "data\\raw\\final.csv"
    df =load_data(data_path)
    
# splitting the data in training and testing set
    x = df.drop('price',axis=1)
    y = df['price']
    split_data(x,y)
    
    xtrain_scaled, xtest_scaled, ytrain, ytest = split_data(x, y) 
    
    print("Linear Regression Model:")
    Linear_Regression(xtrain_scaled, xtest_scaled, ytrain, ytest)
    
    
    print("DecisionTreeRegressor:")
    Decision_TreeRegressor(xtrain_scaled,xtest_scaled,ytrain,ytest)
    
    
    print("Random Forest Model:")
    RandomForest(xtrain_scaled, xtest_scaled, ytrain, ytest)