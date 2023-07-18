import os, sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,  mean_absolute_error, mean_squared_error
from src.utils import save_object
from src.utils import evaluate_model
from dataclasses import dataclass

class ModelTrainerConfig:
    model_path: str= os.path.join('artifact', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.modeltrainerconfig= ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            model={'LinearRegression':LinearRegression()}
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,model)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = model[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.modeltrainerconfig.model_path,
                 obj=best_model
            )


        except Exception as e:
            logging.info("Exception occur at Model Training Stage")
            raise CustomException(e, sys)
            

