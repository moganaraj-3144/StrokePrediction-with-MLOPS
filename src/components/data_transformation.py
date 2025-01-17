import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformationConfig = DataTransformationConfig()

    def get_transformation_obj(self):

        try:
            num_variables = ['age', 'bmi', 'avg_glucose_level']
            cat_variables = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info(f"Numerical variables are: {num_variables}")
            logging.info(f"Categorical variables are: {cat_variables}")

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",  SimpleImputer(strategy="most_frequent")),
                    ("encoding", OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipe", num_pipeline, num_variables),
                    ("cat_pipe", cat_pipeline, cat_variables)
                ]
            )
            logging.info("Pipeline completed successfully")

            return preprocessor
        

        except Exception as e:
            raise CustomException(e,sys)

    
    def initiate_dataTransf(self, train_path, test_path):
        logging.info("DataTransformation started")

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Remove rows with gender as 'Other' in train and test sets
            train_df = train_df[train_df['gender'] != 'Other']
            test_df = test_df[test_df['gender'] != 'Other']
            
            # Replace the num values in 'hypertension' and 'heart_disease' columns to yes/no
            binary_columns = ['hypertension', 'heart_disease']
            for col in binary_columns:
                train_df[col] = train_df[col].replace({1 : 'Yes', 0 : 'No'})
                test_df[col] = test_df[col].replace({1 : 'Yes', 0 : 'No'})

            
            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_transformation_obj()

            target_col = "stroke"

            X_train = train_df.drop(columns=["id",target_col], axis=1)
            y_train = train_df[target_col]
            X_test = test_df.drop(columns=["id",target_col], axis=1)
            y_test = test_df[target_col]


            logging.info("Handling class imbalance using RandomOverSampler")

            if y_train.nunique() <= 2:  # Binary classification
                oversampler = RandomOverSampler(random_state=42)
                logging.info(f"Original class distribution: {y_train.value_counts()}")
                X_train, y_train = oversampler.fit_resample(
                    X_train, y_train
                )
                logging.info(f"Balanced class distribution: {y_train.value_counts()}")

                logging.info("Upsampling completed. Class distribution balanced.")

            logging.info("Applying transformation on train and test sets")

            X_train_arr = preprocessor_obj.fit_transform(X_train)
            X_test_arr = preprocessor_obj.transform(X_test)

            train_arr = np.c_[X_train_arr, np.array(y_train)] 

            test_arr = np.c_[X_test_arr, np.array(y_test)]

            save_object(
                file_path = self.transformationConfig.preprocessor_obj_path,
                obj = preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.transformationConfig.preprocessor_obj_path,
            )

        except Exception as e:
            raise CustomException(e,sys)





















    

