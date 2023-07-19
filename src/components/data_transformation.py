import os, sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_file_path:str = os.path.join('artifact', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformationconfig=DataTransformationConfig()

    def get_preprocessor_object(self):
        try:

           pipeline= Pipeline(steps= [("Standard Scaling", StandardScaler())])

           
           return pipeline
        
        except Exception as e:
            logging.info("Exception occur in preprocessor stage")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self,train_data, test_data):
        try:

            logging.info("Starting Data Transformation")
            train_df= pd.read_csv(train_data)
            test_df= pd.read_csv(test_data)
            x_train= train_df[['indus', 'nox', 'rm', 'tax', 'ptratio', 'lstat', 'dis' , 'age' ]]
            x_test = test_df[['indus', 'nox', 'rm', 'tax', 'ptratio', 'lstat', 'dis' , 'age' ]]
            
            y_train_df= train_df['medv']
            y_test_df= test_df['medv']
          
            preprocessor_obj= self.get_preprocessor_object()

            preprocessor_obj.fit(x_train)
            x_preprocessed_train= preprocessor_obj.transform(x_train)
            x_preprocessed_test = preprocessor_obj.transform(x_test)

            train_arr = np.c_[x_preprocessed_train, np.array(y_train_df)]
            test_arr = np.c_[x_preprocessed_test, np.array(y_test_df)]

            save_object(

                file_path=self.transformationconfig.preprocessor_file_path,
                obj=preprocessor_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.transformationconfig.preprocessor_file_path
            )
        except Exception as e:
            logging.info("Exception occur in initiate data transformation stage")
            raise CustomException(e, sys)
    