import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from bikeshare_model import __version__ as _version
from bikeshare_model.config.core import config
from bikeshare_model.pipeline import bikeshare_pipe
from bikeshare_model.processing.data_manager import load_pipeline

pipeline_file_name = f"{config.app_config.pipeline_save_file}_{_version}.pkl"
bikeshare_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    input_df = pd.DataFrame(input_data)
    """Make a prediction using a saved model """
    validated_data=input_df.reindex(columns=config.model_config.features)
    results = {"predictions": None, "version": _version}
    
    predictions = bikeshare_pipe.predict(validated_data)

    results = {"predictions": predictions,"version": _version}
    print(results)

    return results

if __name__ == "__main__":

    data_in={
        'dteday':['2012-11-05'],
        'season':['winter'],
        'hr':['6am'],
        'holiday':['No'],
        'weekday':['Mon'],
        'workingday':['Yes'],
        'weathersit':['Mist'],
        'temp':[6.1],
        'atemp':[3.0014000000000003],
        'hum':[49.0],
        'windspeed':[19.0012]
    }
    
    make_prediction(input_data=data_in)
