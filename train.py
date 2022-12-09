import bentoml
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATASET_PATH = "dataset.csv"

def load_dataset():
    '''
    Load dataset
    '''
    # load white wine data
    white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')
    # load red wine data
    red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')
    # Add type column to 'red' with value
    red['type'] = 1
    # Add type column to 'white' with value 0
    white['type'] = 0
    # Append datasets to a new dataset called wines
    dataset = pd.concat([red, white], ignore_index=True)

    return dataset

def split_dataset(df):
    '''
    Split in X and Y in train and test
    '''
  # Specify X and y
    X = df.drop(['quality'],axis=1)
    y = df['quality']

    # Split the data up in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    return X_train, X_test, y_train, y_test

def scale_dataset(X_train, X_test):
    '''
    Scale dataset
    '''
    # Scale dataset
    standard_scaler = StandardScaler()
    # Scale the train set
    X_train = pd.DataFrame(standard_scaler.fit_transform(X_train), columns=X_train.columns)
    # Scale the test set
    X_test = pd.DataFrame(standard_scaler.transform(X_test), columns=X_test.columns)

    return standard_scaler, X_train, X_test

def hyperparameter_tuning(X_train, y_train):
    '''
    Model and Optimize the performance using a Grid-search and
    Cross-validation strategy to search the best configurations (model and hyperparameters)
    '''
    # Create a dummy regressor for building the pipeline
    pipe = Pipeline([('regressor', DummyRegressor())])

    # Create space of candidate models and some of their hyperparameters
    search_space = [{'regressor': [LinearRegression()]},
                    {'regressor': [RandomForestRegressor(random_state=0)],
                    'regressor__n_estimators': [10, 50, 100, 200],
                    'regressor__max_depth': [None, 5, 10, 15]}]

    # Search the space in a grid-like approach, selecting at the end the model with the best RMSE metric
    gs = RandomizedSearchCV(pipe, search_space,
                            cv=5, verbose=2,
                            scoring='neg_root_mean_squared_error')
    best_model = gs.fit(X_train, y_train)

    return best_model

def performance_test(model, y_test):
    # Check the best model performance
    y_pred = model.predict(X_test)
    # round and clip predictions to the possible values for quality
    y_pred = y_pred.round().clip(0, 10)
    # RMSE
    rmse = mean_squared_error(y_pred, y_test, squared=False)
    print(rmse)

    return rmse

def save_bento(model,scaler, rmse):
    saved_model = bentoml.sklearn.save_model(
        "wine",   # model name in the local model store
        model.best_estimator_,  # model instance being saved
        labels={    # user-defined labels for managing models in Yatai
            "owner": "cpereira",
            "stage": "dev",
        },
        metadata={  # user-defined additional metadata
            "rmse": rmse,
            "dataset_version": "20221101",
        },
        custom_objects={    # save additional user-defined python objects
            "scaler": scaler,
        }
    )
    print(f"Model saved: {saved_model}")


if __name__ == "__main__":
    # load dataset
    dataset = load_dataset()
    # split dataset in train and test
    X_train, X_test, y_train, y_test = split_dataset(dataset)
    # scale dataset
    scaler, X_train, X_test = scale_dataset(X_train, X_test)
    # model and tune
    best_model = hyperparameter_tuning(X_train, y_train)
    # get performance in test dataset
    rmse = performance_test(best_model, y_test)
    # save in bentoml
    save_bento(best_model, scaler, rmse)
