# Basic Import
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet
from xgboost import XGBRegressor
#from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_models

from dataclasses import dataclass
import sys
import os

from urllib.parse import urlparse
import mlflow
import mlflow.sklearn



@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Split training and test input data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
            "LinearRegression":LinearRegression(),
            "Lasso":Lasso(),
            "Ridge":Ridge(),
            "Elasticnet":ElasticNet(),
            "DecisionTree":DecisionTreeRegressor(),
            "XGBRegressor": XGBRegressor()
             }
            
            params={
                "DecisionTree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },

                "Elasticnet":{
                    'alpha': [0.01, 0.1, 1.0, 10.0],  # Range of alpha values (regularization strength)
                    'l1_ratio': [0.1, 0.5, 0.7, 0.9],  # Mixing parameter (0 for L2, 1 for L1)
                    'fit_intercept': [True, False]
                },

                "Lasso":{
                    'alpha': [0.01, 0.1, 1.0, 10.0],  # Range of alpha values (regularization strength)
                    'fit_intercept': [True, False]
                },

                "LinearRegression":{},

                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },


                "Ridge":{
                    'alpha': [0.01, 0.1, 1.0, 10.0],  # Range of alpha values (regularization strength)
                    'fit_intercept': [True, False],     # Include or exclude intercept in the model
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']  # Solver options
                }
            }
            
            #Utils function  use..
            model_report:dict=evaluate_models(X_train,y_train,X_test,y_test,models,params)
            print(model_report)
            print('\n======================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print("This is the best model:")
            print(best_model_name)

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')


            #MLOPS start for tracking
            model_names = list(params.keys())

            actual_model=""

            for model in model_names:
                if best_model_name == model:
                    actual_model = actual_model + model

            best_params = params[actual_model]

            mlflow.set_registry_uri("https:")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


            # mlflow

            with mlflow.start_run():

                predicted_qualities = best_model.predict(X_test)

                (rmse, mae, r2) = self.eval_metrics(y_test, predicted_qualities)

                mlflow.log_params(best_params)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)


                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=actual_model)
                else:
                    mlflow.sklearn.log_model(best_model, "model")



            
            
            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
        

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)