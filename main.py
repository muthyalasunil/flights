# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import datetime
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

DATA_FOLDER = 'data'


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    # print(f'Hi, {name} today is ')  # Press Ctrl+F8 to toggle the breakpoint.
    print('Hi %s, today is: %s' % (name, datetime.datetime.today().strftime("%Y / %m /%d")))


def calculate(number1, number2):
    product = number1 * number2
    print('the product of %s and %s is %s' % (number1, number2, product))


def load_data(filename):
    airline_data_df = pd.read_csv(DATA_FOLDER + '\\' + filename)
    return airline_data_df

def test_train(flights_df):

    # Labels are the values we want to predict
    labels = np.array(flights_df['DELAY_FLAG'])
    # Remove the labels from the features
    # axis 1 refers to the columns
    flights_df = flights_df.replace(np.nan, 0)
    features = flights_df.drop(['ARRIVAL_DELAY', 'CANCELLATION_REASON', 'DIVERTED', 'CANCELLED', 'DELAY_FLAG'], axis=1)

    pickle.dump(features.columns, open('flights_df_model.features', 'wb'))

    # Saving feature names for later use
    feature_list = list(features.columns)
    print(feature_list)

    # Convert to numpy array
    features = np.array(features)
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                                random_state=42)
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    # Instantiate model with 1000 decision trees
    rf_model = RandomForestClassifier(n_estimators=1000)
    rf_model.fit(train_features, train_labels)

    filename = 'rf_flights_model.sav'
    pickle.dump(rf_model, open(filename, 'wb'))
    y_pred = rf_model.predict(test_features)
    print("RF Accuracy:", metrics.accuracy_score(test_labels, y_pred))
    confusion_matrix = pd.crosstab(test_labels, y_pred, rownames=['Actual'], colnames=['Predicted'])
    print(confusion_matrix)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    '''
    airline_df = load_data('airlines.csv')
    airline_df = airline_df.sort_values(by=['AIRLINE'])
    print(airline_df)

    airport_df = load_data('airports.csv')
    airport_df = airport_df.sort_values(by=['AIRPORT'])
    df_first_10 = airport_df.iloc[:10]
    print(df_first_10[['CITY', 'AIRPORT']])
    '''

    flights_df = load_data('flights.csv')
    flights_df['DELAY_FLAG'] = np.where(flights_df['ARRIVAL_DELAY'] > 0, 1, 0)
    # storing in new variable
    # display
    #print(flights_df.columns.values)
    #print(flights_df.iloc[:10].to_string())

    #AIRLINE, TAIL_NUMBER, ORIGIN_AIRPORT, DESTINATION_AIRPORT
    arr_airline = flights_df.AIRLINE.unique()
    arr_tail_nbr = flights_df.TAIL_NUMBER.unique()
    arr_orig_airport = flights_df.ORIGIN_AIRPORT.unique()
    arr_dest_airport = flights_df.DESTINATION_AIRPORT.unique()
    map_values = dict(zip(arr_airline, range(len(arr_airline))))
    flights_df["AIRLINE"] = flights_df["AIRLINE"].apply(lambda x: map_values[x])
    map_values = dict(zip(arr_tail_nbr, range(len(arr_tail_nbr))))
    flights_df["TAIL_NUMBER"] = flights_df["TAIL_NUMBER"].apply(lambda x: map_values[x])
    map_values = dict(zip(arr_orig_airport, range(len(arr_orig_airport))))
    flights_df["ORIGIN_AIRPORT"] = flights_df["ORIGIN_AIRPORT"].apply(lambda x: map_values[x])
    map_values = dict(zip(arr_dest_airport, range(len(arr_dest_airport))))
    flights_df["DESTINATION_AIRPORT"] = flights_df["DESTINATION_AIRPORT"].apply(lambda x: map_values[x])

    test_train(flights_df)
