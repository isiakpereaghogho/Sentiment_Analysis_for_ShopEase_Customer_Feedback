import pandas as pd

from pathlib import Path
import numpy as np
import logging
import re

from config.constant import Input_Data
logging.basicConfig(level=logging.INFO)

def data_ingestion():
    logging.info("Starting data ingestion...")
    try:
        data = pd.read_csv(Input_Data)
        logging.info(f"Data loaded successfully")
        print(data.head(5))
        return data
    except Exception as e:
        logging.error(f"Error occured while loading the data: {e}")
        
data_ingestion()
