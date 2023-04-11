import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,FunctionTransformer,MinMaxScaler
from sklearn.impute import SimpleImputer


class Pipeline:
    def __init__(self):
        self.pipeline = None
        
    
    #@staticmethod
    def cyclical_encoding(self,x,x_max):
        cos_x=np.cos(2*np.pi*x /(x_max+1))
        sin_x=np.sin(2*np.pi*x /(x_max+1))
        return np.c_[cos_x,sin_x]
        
     
      
    def fit(self, X, y):

        # data preparation
        X = self.data_preparation(X)
        
        # model training
        self.pipeline = SKPipeline(steps=[
            ('preprocessor', ColumnTransformer(
                transformers=[
                   #('normalize',MinMaxScaler(),['hum','temp','windspeed']),
                   #('ordinal',OrdinalEncoder(), ['mnth']),
                   ('cyclical',FunctionTransformer(self.cyclical_encoding,kw_args={'x_max':6}), ['weekday']),
                   #('onehot',OneHotEncoder(), ['workingday','holiday']),  
                    
                ],
                remainder='passthrough')),
            ('regressor', GradientBoostingRegressor(loss='absolute_error',n_estimators=1000,random_state=34))
        ])
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        
        X=self.data_preparation(X)
        return self.pipeline.predict(X)


    def data_preparation(self, X):
        #drop any null values
        X=X.dropna()
        #drop duplicates
        X = X.drop_duplicates()

        #Change the datatypes of columns if they are not in the required type. For example, dteday column if it is not date type, 
        # It can be changed to date datatype as follows.
        X['dteday']=pd.to_datetime(X['dteday'])

        # feature selection
        X = X[['yr','mnth','weekday','workingday','holiday','hum','temp','windspeed']]
        return X

    
