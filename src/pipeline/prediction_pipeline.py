import os, sys
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object
import pandas as pd

class PredictOutput :
    def __init__(self):
        pass

    def get_output(self, df):
        
      try:
         logging.info("Predicting the output and loading object ")
         preprocessor_obj= load_object(os.path.join('artifact','preprocessor.pkl'))
         model= load_object(os.path.join('artifact','model.pkl'))
    
         logging.info("Transforming the input dataframe ")
         df_transform= preprocessor_obj.transform(df)
         y_pred= model.predict(df_transform)

         return y_pred

      except Exception as e:
        raise CustomException(e, sys)

class CustomData:
    def __init__(self, indus:float,
                 nox:float,
                 rm:float,
                 tax:float,
                 ptratio:float,
                 lstat:float,
                 dis:float,
                 age:float ) :
                
        self.indus= indus
        self.nox=nox
        self.rm=rm
        self.tax= tax
        self.ptratio=ptratio
        self.lstat=lstat
        self.dis=dis
        self.age= age
        
    ## ['indus', 'nox', 'rm', 'tax', 'ptratio', 'lstat', 'dis' , 'age' ]
    
    def get_data_as_dataframe(self):
        try: 
            custom_data_input_dict = {
                'indus':[self.indus],
                'nox':[self.nox],
                'rm':[self.rm],
                'tax':[self.tax],
                'ptratio':[self.ptratio],
                'lstat':[self.lstat],
                'dis':[self.dis],
                'age':[self.age]
            }

            df = pd.DataFrame(custom_data_input_dict)
            return df
        except Exception as e:
            raise CustomException(e, sys)