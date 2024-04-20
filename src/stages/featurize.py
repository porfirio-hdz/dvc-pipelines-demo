import argparse
import pandas as pd
from typing import Text
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src.utils.logs import get_logger


def renaming_fun(x):
    if "remainder__" in x:
        return x.strip('remainder__')
    return x

def featurize(config_path: Text) -> None:
    """Create new features.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('FEATURIZE', log_level=config['base']['log_level'])

    logger.info('Load raw data')
    dataset = pd.read_csv(config['data_load']['dataset_prepare'])

    # Drop columns
    cols_to_drop = config['featurize']['cols_to_drop']
    X = dataset.drop(cols_to_drop, axis=1)

    logger.info('Extract features')
    # Define categorical features
    categorical_features = config['featurize']['categorical_features']

    # Create a column transformer with one-hot encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_features)
        ],
        remainder='passthrough'
    )

    X_processed = preprocessor.fit_transform(X)
    # Convert processed X array into dataframe
    X_processed_df = pd.DataFrame(X_processed, columns=preprocessor.get_feature_names_out())

    # Clean column names
    X_processed_df.columns = [renaming_fun(col) for col in X_processed_df.columns]

    logger.info('Save features')
    features_path = config['featurize']['features_path']
    X_processed_df.to_csv(features_path, index=False)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    featurize(config_path=args.config)