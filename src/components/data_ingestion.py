import os, sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    file_path :str= os.path.join('data', 'boston.csv')
    train_data_path:str= os.path.join('artifact','train_data.csv')
    test_data_path :str= os.path.join('artifact','test_data.csv')

class DataIngestion:
    def __init__(self):
        self.dataingestionconfig= DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        try :
            logging.info("Starting Data Ingestion")
            data= pd.read_csv(self.dataingestionconfig.file_path ,index_col='Unnamed: 0')
            
            train_data, test_data= train_test_split(data, test_size=0.25)
            
            os.makedirs(os.path.dirname(self.dataingestionconfig.train_data_path), exist_ok=True)

            logging.info("Splitting the data as train and test")

            train_data.to_csv(self.dataingestionconfig.train_data_path,index=False,header=True)
            test_data.to_csv(self.dataingestionconfig.test_data_path,index=False,header=True)
            
            logging.info("Data Ingestion Completed")
            return (self.dataingestionconfig.train_data_path, self.dataingestionconfig.test_data_path)

        except Exception as e:
            logging.info("Exception occur in data ingestion")
            raise CustomException(e, sys)
        


if __name__=="__main__":
    obj= DataIngestion()
    obj.initiate_data_ingestion()