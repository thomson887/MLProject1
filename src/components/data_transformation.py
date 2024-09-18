import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer #to create a pipeline of processes
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformer_object(self):
        '''
        This function creates the pipeline for transformation and then defines the preprocessor object which has the column transformation.
        '''
        try:
            numerical_columns = ['writing score', 'reading score']
            categorical_columns = [
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course'
            ]
            num_pipeline=Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            cat_pipeline=Pipeline(
                #The reason that adding with_mean=False resolved my error is that the StandardScaler is subtracting the mean from each feature, which can result in some features having negative values. However, StandardScaler() alone assumes that the features have positive values, which can cause issues when working with features that have negative values.
                #By setting with_mean=False, the StandardScaler does not subtract the mean from each feature, and instead scales the features based on their variance. This can help preserve the positive values of the features and avoid issues.
                
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info(f'Categorical columns: {categorical_columns}')
            logging.info(f'Numerical columns: {numerical_columns}')

            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipelines',cat_pipeline,categorical_columns)
                ]
            )
            logging.info('preprocessor object has been created successfully.')

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def initiate_data_transformation(self, train_path, test_path):
        '''
        Takes the training and test data from the train path and test path and then calls the function get_data_transformer_object() which performs column transformation in a pipeline.
        '''
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data successfully')
            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformer_object() #calls the get_data_transformer_object

            target_column_name = 'math score'
            numerical_columns = ['writing score', 'reading score']

            #defining train dataset input and target features
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            #Defining test dataset input and target features.
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info('Applying preprocessing object on training dataframe and test dataframe.')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df) #fit-transform on training data the preprocessing steps defined in get_data_transformer_object()
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df) # Only transform on test data.

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ] 
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info('Saved preprocessing object.')
            
            #Saving the preprocessor object into a pickle file into the location by calling the save_object function and giving the file path and preprocessing object as the parameters.
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
           
        except Exception as e:
            raise CustomException(e,sys)



                         
      

