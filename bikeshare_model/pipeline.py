import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from bikeshare_model.config.core import config
from bikeshare_model.processing.features import Mapper, one_hot_encoder, weekday_imputter, weathersit_imputter, get_year_and_month, handle_outliers

hour_mapping = {'4am': 0, '3am': 1, '5am': 2, '2am': 3, '1am': 4, '12am': 5, '6am': 6, '11pm': 7, '10pm': 8,
                '10am': 9, '9pm': 10, '11am': 11, '7am': 12, '9am': 13, '8pm': 14, '2pm': 15, '1pm': 16,
                '12pm': 17, '3pm': 18, '4pm': 19, '7pm': 20, '8am': 21, '6pm': 22, '5pm': 23}
workingday_mapping = {'No': 0, 'Yes': 1}
holiday_mapping = {'Yes': 0, 'No': 1}
weather_mapping = {'Heavy Rain': 0, 'Light Rain': 1, 'Mist': 2, 'Clear': 3}
season_mapping = {'spring': 0, 'winter': 1, 'summer': 2, 'fall': 3}
mnth_mapping = {'January': 0, 'February': 1, 'December': 2, 'March': 3, 'November': 4, 'April': 5,
                'October': 6, 'May': 7, 'September': 8, 'June': 9, 'July': 10, 'August': 11}
yr_mapping = {2011: 0, 2012: 1}

bikeshare_pipe = Pipeline([
    
    ##==Transform Date, this adds mnth and yr==##
    ("date_transform", get_year_and_month(variables=config.model_config.dteday_var) ),
    ##==Imputter==##
    ("weekday_imputter", weekday_imputter(variables=config.model_config.weekday_var) ),
    ("weathersit_imputter", weathersit_imputter(variables=config.model_config.weathersit_var) ),
    ##==========Mapper======##
    ("map_hr", Mapper(config.model_config.hr_var, hour_mapping) ),
    ("map_workingday", Mapper(config.model_config.workingday_var, workingday_mapping ) ),
    ("map_holiday", Mapper(config.model_config.holiday_var, holiday_mapping) ),
    ("map_weathersit", Mapper(config.model_config.weathersit_var, weather_mapping) ),
    ("map_season", Mapper(config.model_config.season_var, season_mapping) ),
    ("map_mnth", Mapper(config.model_config.mnth_var, mnth_mapping) ),
    ("map_yr", Mapper(config.model_config.yr_var, yr_mapping) ),
    ##==Handle Outliers==##
    ("temp_handle_outliers", handle_outliers(config.model_config.temp_var) ),
    ("atemp_handle_outliers", handle_outliers(config.model_config.atemp_var) ),
    ("hum_handle_outliers", handle_outliers(config.model_config.hum_var) ),
    ("windspeed_handle_outliers", handle_outliers(config.model_config.windspeed_var) ),
    ##==One Hot Encoder==##
    ("ohe_weekday", one_hot_encoder(config.model_config.weekday_var) ),
    ##==ML Algo==##
    # 91%
    # ('model_rf', RandomForestRegressor(n_estimators=config.model_config.n_estimators,
    #                                     max_depth=config.model_config.max_depth,
    #                                     random_state=config.model_config.random_state))
    # 94%
    ('model_XGB', XGBRegressor(n_estimators=config.model_config.n_estimators,
                               max_depth=config.model_config.max_depth,
                               random_state=config.model_config.random_state)
    )
])