import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass #used to create class variables

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass #A decorator that automatically adds special methods to the class (like __init__, __repr__, etc) based on the class attributes (in this case attributes are train_data_path..).
class DataIngestionConfig:
    """Any data which is required will be obtained using this"""
    train_data_path: str=os.path.join('artifacts', 'train.csv') #path where training data will be saved inside an artifacts folder.
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        """Initializes an instance of the dataIngestionConfig class so that the file paths for storing the data are available in self.ingestion_config"""
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(r'C:\Users\justi\OneDrive\STUDY\Python\MLProject1\Data\StudentsPerformance.csv')
            logging.info('Read the dataset as a dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("train test split initiated")
            # Can write this train test split code in utils.py so that it can be called again later.
            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Ingestion of the data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    obj=DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()

    #data transformation
    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    #model training
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
