'''
The script takes the database file path and model file path, creates and trains a classifier, and stores the classifier
into a pickle file to the specified model file path.

This file:
 Loads data from the SQLite database
 Splits the dataset into training and test sets
 Builds a text processing and machine learning pipeline that uses NLTK, scikit-learn's Pipeline and GridSearchCV
 Trains and tunes a model using GridSearchCV
 Outputs results on the test set: a final model that uses the message column to predict classifications for 36 categories
 Exports the final model as a pickle file
'''

# import libraries
import sys
from sqlalchemy import create_engine
import pandas as pd
import re
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

class Descr_df(object):
    """
    TODO
    """
    def transform (self, X):
        """
        Display some features of the dataset
        """
        #print ("Structure of the data: \n {}".format(X.head(5)))
        #print ("Features names: \n {}".format(X.columns))
        print ("Shape of the data: \n {}".format(X.shape))
        return X

    def fit(self, X, y=None):
        return self

class Fillna(object):
    """
    TODO
    """
    def transform(self, X):
        """
        fillna with 0 (from dummy columns with 0)
        """

        X = X.fillna(0)
        return X

    def fit(self, X, y=None):
        return self

def load_database(database_filepath):
    """
    This function load the message database.

    :param database_filepath: filepath of the disaster messages database
    :return: X: text message column
             y: categories
             category_names: name of each category column
    """
    # read clean data pickle
    df = pd.read_pickle(database_filepath)

    featured_col = ['Num Voters', 'Num Players Min', 'Num Players Max', 'Playtime Min', 'Playtime Max', 'Weight', 'Player Min Age', \
        'Year_Num', 'Extensive', 'Moderate', 'No', 'Abstract', 'Children\'s', 'Customizable', 'Family', 'Party', 'Strategy', 'War', 'Thematic']

    X = df[featured_col]
    y = df['Avg Rating']

    return X, y, featured_col


def build_regression_model(model_name):
    """ Build the pipeline MultipleLinearRegression including
    filling na values and implementing a grid search.

    pipeline : the custom made pipeline
    is_grid_search_present : flag showing if a grid search is done
    """

    pipeline = Pipeline(steps=[
        #('DataframeDescription', Descr_df()),
        ('FillNa', Fillna()),
        (model_name, LinearRegression())
    ])

    is_grid_search_present = False
    return pipeline, is_grid_search_present

def build_decision_tree_model(model_name):
    """ Build the pipeline DecisionTreeRegressor including
    filling na values and implementing a grid search.

    pipeline : the custom made pipeline
    is_grid_search_present : flag showing if a grid search is done
    """

    pipeline = Pipeline(steps=[
        #('DataframeDescription', Descr_df()),
        ('FillNa', Fillna()),
        ('dec_tree', DecisionTreeRegressor())
    ])

    # tune the model
    parameters = {
        'dec_tree__max_depth': [2, 4, 6, 8],
    }
    pipeline = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=1)

    is_grid_search_present = True
    return pipeline, is_grid_search_present

def build_random_forest_regressor_model(model_name):
    """ Build the pipeline RandomForestRegressor including
    filling na values and implementing a grid search.
    
    Args:
        None

    Returns:
        pipeline: the custom made pipeline
        is_grid_search_present: flag showing if a grid search is done
    """

    pipeline = Pipeline(steps=[
        #('DataframeDescription', Descr_df()),
        ('FillNa', Fillna()),
        #('scaler', StandardScaler()),
        ('forest_regressor', RandomForestRegressor(random_state=11))
    ])

    # tune the model (model_name__parameter)
    parameters = [{
        'forest_regressor__min_samples_leaf': [1,2,3,10,15],
        'forest_regressor__max_depth': [10, 20, 30],
        'forest_regressor__n_estimators': [500,1500,2500]
    }]
    pipeline = GridSearchCV(pipeline, param_grid=parameters, return_train_score=True, n_jobs=-1)

    is_grid_search_present = True
    return pipeline, is_grid_search_present

def evaluate_model(model, X_test, Y_test, input_col):
    """
    This function shows the accuracy, precision, and recall of the tuned model.
    For regression machine learning the two most commonly used metrics are the 
    Mean Squared Error (MSE) and the Mean Absolute Error (MAE).
    """
    Y_pred = model.predict(X_test)

    # calculate the Mean Squared Error (MSE)
    # MSE penalizes large errors.
    mse = mean_squared_error(Y_pred, Y_test)
    print("Mean Square Error is {}".format(np.round(mse, 3)))

    # calculate the Mean Squared Error (MSE)
    # MSE penalizes large errors.
    mae = mean_absolute_error(Y_pred, Y_test)
    print("Mean Absolute Error is {}".format(np.round(mae, 3)))
   

def print_coeficients(model, model_name, input_col):
    """
    """
    try:
        # pair the feature names with the coefficients
        coefs = list(zip(input_col, model.named_steps[model_name].coef_))
        coefs.sort(key= lambda x:x[1], reverse=True)
        print(coefs)
    except:
        print('Couldn\'t get coeficients - {} '.format(model_name))


def get_cv_scores(model, X_train, y_train):
    """
    function to get cross validation scores
    The low R² value indicates that our model is not very accurate. 
    The standard deviation value indicates we may be overfitting the training data.
    """
    scores = cross_val_score(model, X_train, y_train, cv = 5, scoring="r2", n_jobs=-1)
    print('CV Mean: ', np.mean(scores))
    print('STD: ', np.std(scores))


def save_model(model, model_filepath):
    """
    This function exports the model as pickle file

    :param model: model to be saved
    :param model_filepath: path to save
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)



model_dict = {
    #'LinearRegression': build_regression_model,
    #'DecisionTree': build_decision_tree_model,
    'RandomForest': build_random_forest_regressor_model
}

# load and prepare data for modelling 
database_filepath = 'Boardgame\\data\\boardgame_data_clean.pkl'
X, Y, features = load_database(database_filepath)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

for model_name in model_dict:
    print('Building {} model...'.format(model_name))
    model, is_grid_search_present = model_dict[model_name](model_name)
    
    print('Training model...')
    #get_cv_scores(model, X_train, Y_train)
    grid_result = model.fit(X_train, Y_train)

    if(is_grid_search_present):
        print('Best Score: ', grid_result.best_score_)
        print('Best Params: ', grid_result.best_params_)

    print('Evaluating model...')
    print_coeficients(model, model_name, features)
    evaluate_model(model, X_test, Y_test, features)
    print('Done')



""" def main():
    if len(sys.argv) == 0:
        #database_filepath, model_filepath = sys.argv[1:]
        database_filepath = 'Boardgame\\data\\boardgame_data_clean.pkl'
        X, Y, features = load_database(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()

 """

