import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from bikeshare_model.config.core import config
from bikeshare_model.processing.features import one_hot_encoder, weekday_imputter, get_year_and_month
from bikeshare_model.processing.data_manager import _load_raw_dataset

sample_input = _load_raw_dataset(file_name=config.app_config.training_data_file)

def test_one_hot_encoder():
    # Given
    transformer = one_hot_encoder(
        variables=config.model_config.weekday_var
    )
    df = sample_input.copy()
    df = df[df['weekday'].notnull()]
    test_idx = df[df['weekday'] == 'Fri'].head(1).index

    # When
    subject = transformer.fit(df).transform(df)

    # Then
    # Test new columns added
    col_list = subject.columns
    exp_list = ['weekday_Mon', 'weekday_Tue', 'weekday_Wed', 'weekday_Thu', 'weekday_Fri', 'weekday_Sat', 'weekday_Sun']
    assert set(col_list).intersection(set(exp_list)) == set(exp_list)
    
    # Test encoding for Friday
    assert subject.loc[test_idx, 'weekday_Fri'].any() == 1
    assert subject.loc[test_idx, 'weekday_Sat'].any() == 0

def test_weekday_imputter():
    #Given
    transformer = get_year_and_month(
        variables=config.model_config.dteday_var
    )
    imputter = weekday_imputter(
        variables=config.model_config.weekday_var
    )
    df = sample_input.copy()
    df = df[df['weekday'].isnull()]
    test_idx = df[df['dteday'] == '2012-06-17'].head(1).index
    df = transformer.fit(df).transform(df)
    
    #When
    subject = imputter.fit(df).transform(df)
    
    #Then
    assert subject.loc[test_idx, 'weekday'].to_list()[0] == 'Sun'

# test_one_hot_encoder()
# test_weekday_imputter()