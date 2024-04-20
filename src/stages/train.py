import argparse
import joblib
import pandas as pd
from typing import Text
import yaml
from sklearn.ensemble import RandomForestClassifier
from dvclive import Live

#from src.train.train import train
from src.utils.logs import get_logger


def train_model(config_path: Text) -> None:
    """Train model.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('TRAIN', log_level=config['base']['log_level'])

    logger.info('Get estimator name')
    estimator_name = config['train']['estimator_name']
    logger.info(f'Estimator: {estimator_name}')

    logger.info('Load train dataset')
    train_df = pd.read_csv(config['data_split']['trainset_path'])

    # Split the data into features (X) and target variable (y)
    target_column = config['featurize']['target_column']
    X = train_df.drop([target_column], axis=1)
    y = train_df[target_column]

    logger.info('Train model')
    # Create model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Fit the model to the training data
    model.fit(X, y)

    logger.info('Save model')
    models_path = config['train']['model_path']
    joblib.dump(model, models_path)


    with Live() as live:
        live.log_artifact(
            str(models_path),
            type="model",
            name="penguins_classifier",
            desc="This is a penguins classifier.",
            labels=["cv", "classification"]
                    #, params.train.arch],
        )


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train_model(config_path=args.config)
