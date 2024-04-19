# dvc-pipelines-demo
 MLOps Bootcamp Module 8 - Turning Notebooks into Pipelines DEMO
 
This repo is intended to instruct how to build a machine-learning pipeline with DVC. The main goal of the repo is to transition from working in a Jupyter notebook to a more modular workflow.

As a first step, it is **recommended** to create a virtual environment with a tool such as
[virtualenv](https://virtualenv.pypa.io/en/stable/):

```console
$ python -m venv .mlops
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

The step above will ensure we have the correct versions of [DVC](https://dvc.org/) and [DVCLive](https://dvc.org/doc/dvclive).

### Initialize the repo with DVC init

Run ```dvc init``` to initialize the DVC repository. This command will create several files related to DVC. 

It is recommended to commit the changes to git before continuing.

### Create the first pipeline stage with DVC

The ```simple_ml_workflow.ipynb``` jupyter notebook located in the ```notebooks``` folder is divided into different sections or steps, which we will use to build each pipeline stage. For example, the first section is the **Data Prepare** part. So, this will be our first pipeline stage.

So we will take the code cell below:

```python
# Load the Penguins dataset
df = pd.read_csv(config['data_load']['dataset_csv'])

# Display the first few rows of the dataset
df.head()
print(df.shape)
# drop NaNs
df = df.dropna(axis=0, how='any')
df = df.drop('Unnamed: 0', axis=1)
```

And transform it into a Python script like this:

```python
import argparse
import pandas as pd
from typing import Text
import yaml

from src.utils.logs import get_logger


def data_load(config_path: Text) -> None:
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('DATA_LOAD', log_level=config['base']['log_level'])

    logger.info('Get dataset')
    # Load the Penguins dataset
    df = pd.read_csv(config['data_load']['dataset_csv'])

    # drop NaNs
    df = df.dropna(axis=0, how='any')
    df = df.drop('Unnamed: 0', axis=1)

    logger.info('Save processed data')
    df.to_csv(config['data_load']['dataset_prepare'], index=False)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_load(config_path=args.config)

```

Using the code above, we will create a Python script called ```data_prepare.py``` and save it in a new ```stages``` folder under the ```src``` folder.

- We can now run a ```dvc status``` command. This will tell us that no pipeline is tracked in the project yet.

Once we have our script, we can create a DVC stage with the following command:

```bash
dvc stage add -n data_prepare \
    -d src/stages/data_prepare.py \
    -o data/processed/prepare_penguins.csv \
    -p base,data_load \
    python -m src.stages.data_prepare --config=params.yaml
```

After running the command above, a ```dvc.yaml``` file is created. You can also check this by running a ```git status```.

To test our first pipeline stage, we can run the command below:

```dvc repro```

If the pipeline runs correctly, you should see a new file created named ```dvc.lock```. This is a state file that DVC creates to capture the pipeline's reproduction results.
- Do not manually modify this file.

Now you can run ```dvc dag``` and see a simple diagram of your first pipeline stage:

![dag_first_stage](/img/dag_data_prepare.png)

> After having a successful stage execution, it is a good practice to commit our changes with Git.

### Create the other pipeline's stages

DVC has two ways of creating pipeline stages. The first is the one we did before using the ```dvc stage add``` command. Remember that after running this command for the first time, DVC creates a ```dvc.yaml``` file that will contain the information of all the stages of our pipeline. 

The second way to create a stage is by editing the ```dvc.yaml``` file.

> However, adding a stage with ```dvc stage add``` has the advantage that it will verify the validity of the arguments provided.

So, to add the second stage (feature processing), we need to create our Python script under the ```src/stages``` directory. We will name this script as ```featurize.py```


```python
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
```

Once the script is under the ```src/stages``` directory, we can add the code below to our ```dvc.yaml``` file (Be aware of proper indentation):

```yaml
  featurize:
    cmd: python -m src.stages.featurize --config=params.yaml
    deps:
    - data/processed/prepare_penguins.csv
    - src/stages/featurize.py
    params:
    - base
    - data_load
    - featurize
    outs:
    - data/processed/featured_penguins.csv
```

> Notice that the output (prepare_penguins.csv) of the data_prepare stage is the input of the featurize stage.

So far, your dvc.yaml file should look like this:

```yaml
stages:
  data_prepare:
    cmd: python -m src.stages.data_prepare --config=params.yaml
    deps:
    - src/stages/data_prepare.py
    params:
    - base
    - data_load
    outs:
    - data/processed/prepare_penguins.csv
  featurize:
    cmd: python -m src.stages.featurize --config=params.yaml
    deps:
    - data/processed/prepare_penguins.csv
    - src/stages/featurize.py
    params:
    - base
    - data_load
    - featurize
    outs:
    - data/processed/featured_penguins.csv
```

We can execute our pipeline again with ```dvc repro```.

Notice again that the ```dvc.lock``` has been modified with the new information from the featurize stage.

And again, you can check your current pipeline with ```dvc dag```.

![dvc_dag_2](/img/dvc_dag_2.png)

> Once again, it is a good practice to commit to our changes.

#### Building the remaining stages

The procedure to build the remaining pipeline stages is similar. We just have to be aware of setting the correct dependencies to build the DAG.

For example to build the next step (data_split) the dvc command is the following:

```bash
dvc stage add -n data_split \
    -d src/stages/data_split.py -d data/processed/featured_penguins.csv \
    -o data/processed/train_penguins.csv -o data/processed/test_penguins.csv \
    -p base,data_load,featurize,data_split \
    python -m src.stages.data_split --config=params.yaml
```

The rest of the stages are left as an exercise. You can download the remaining Python scripts from this [link]().

 For a completed solution you can refer to this [**repo**](https://github.com/porfirio-hdz/dvc-pipelines).

 ### DVC Studio

 Instructions to sign up to [DVC Studio](https://dvc.org/doc/studio) and how to add your first project to DVC Studio can be found [here](https://dvc.org/doc/studio/user-guide/experiments/create-a-project#connect-to-a-git-repository-and-add-a-project).

### More resources

- A complete guide to DVC's model registry features can be found [here](https://dvc.org/doc/start/model-management/model-registry).