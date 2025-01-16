import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.utils import save_object, evaluate_models
from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    model_trainer_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiateModelTrainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "Decision Tree Classifier": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "XGBClassifier": XGBClassifier(),
                "GradientBoost Classifier": GradientBoostingClassifier(),
                
                }
            
            params={
                "K-Neighbors Classifier":{
                    "n_neighbors": [3, 5, 7, 9, 11],
                    
                },
                "Decision Tree Classifier": {
                    'criterion':["gini", "entropy", "log_loss"],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest Classifier":{
                    "n_estimators": [100, 200],
                    "max_depth": [10, 20],
                    "min_samples_split": [2, 5]
                },

                "AdaBoost Classifier":{
                    "n_estimators": [50, 100, 200],  # Number of weak learners
                    "learning_rate": [0.01, 0.1, 0.5, 1.0],
                },             

                "XGBClassifier":{
                    "n_estimators": [100, 200, 300],  # Number of boosting rounds
                    "learning_rate": [0.01, 0.1, 0.2, 0.3],  # Step size shrinkage
                },
                "GradientBoost Classifier":{
                    "n_estimators": [100, 200, 300],  # Number of boosting stages
                    "learning_rate": [0.01, 0.1, 0.2],  # Learning rate shrinks contribution of each tree
                    # "max_depth": [3, 5, 7], 
                    # "min_samples_split": [2, 5, 10], 
                    # "min_samples_leaf": [1, 2, 4], 
                    "subsample": [0.7, 0.8, 1.0],  
                    # "max_features": ["sqrt", "log2", None],  
                }                

            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models, param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.model_trainer_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            acc = accuracy_score(y_test, predicted)
            return acc
            

            
        except Exception as e:
            raise CustomException(e,sys)