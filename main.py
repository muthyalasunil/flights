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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

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
    features = flights_df.drop(['DELAY_FLAG'], axis=1)

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

    log_reg = LogisticRegression(random_state=0, solver='saga', max_iter=1000, multi_class='auto').fit(train_features,
                                                                                                       train_labels)
    filename = 'log_reg_model.sav'
    pickle.dump(log_reg, open(filename, 'wb'))
    y_pred1 = log_reg.predict(test_features)
    print("Log Reg Accuracy:", metrics.accuracy_score(test_labels, y_pred1))
    confusion_matrix = pd.crosstab(test_labels, y_pred1, rownames=['Actual'], colnames=['Predicted'])
    print(confusion_matrix)

    # Instantiate model with 1000 decision trees
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(train_features, train_labels)

    filename = 'rf_flights_model.sav'
    pickle.dump(rf_model, open(filename, 'wb'))
    y_pred2 = rf_model.predict(test_features)
    print("RF Accuracy:", metrics.accuracy_score(test_labels, y_pred2))
    confusion_matrix = pd.crosstab(test_labels, y_pred2, rownames=['Actual'], colnames=['Predicted'])
    print(confusion_matrix)

    knn = DecisionTreeClassifier()
    knn.fit(train_features, train_labels)
    filename = 'dtree_flights_model.sav'
    pickle.dump(knn, open(filename, 'wb'))
    y_pred3 = knn.predict(test_features)
    print("DecisionTreeClassifier Accuracy:", metrics.accuracy_score(test_labels, y_pred3))
    confusion_matrix = pd.crosstab(test_labels, y_pred3, rownames=['Actual'], colnames=['Predicted'])
    print(confusion_matrix)

    # predict probabilities
    pred_prob1 = log_reg.predict_proba(test_features)
    pred_prob2 = rf_model.predict_proba(test_features)
    pred_prob3 = knn.predict_proba(test_features)

    # roc curve for models
    fpr1, tpr1, thresh1 = roc_curve(test_labels, pred_prob1[:,1], pos_label=1)
    fpr2, tpr2, thresh2 = roc_curve(test_labels, pred_prob2[:,1], pos_label=1)
    fpr3, tpr3, thresh3 = roc_curve(test_labels, pred_prob3[:,1], pos_label=1)

    # roc curve for tpr = fpr
    random_probs = [0 for i in range(len(test_labels))]
    p_fpr, p_tpr, _ = roc_curve(test_labels, random_probs, pos_label=1)

    # auc scores
    auc_score1 = roc_auc_score(test_labels, pred_prob1[:,1])
    auc_score2 = roc_auc_score(test_labels, pred_prob2[:,1])
    auc_score3 = roc_auc_score(test_labels, pred_prob3[:,1])

    print(auc_score1, auc_score2, auc_score3)

    plt.style.use('seaborn')

    # plot roc curves
    plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Logistic Regression')
    plt.plot(fpr2, tpr2, linestyle='--',color='green', label='Random Forest Classifier')
    plt.plot(fpr3, tpr3, linestyle='--',color='blue', label='DecisionTree Classifier')
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
    # title
    plt.title('ROC curve')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')

    plt.legend(loc='best')
    plt.savefig('ROC',dpi=300)
    plt.show();


def predict_data(features, print_score=False):

    filename = 'log_reg_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    feature_names = pickle.load(open('flights_df_model.features', 'rb'))

    # summarize feature importance
    if print_score:
        importance = loaded_model.feature_importances_
        for i,v in enumerate(importance):
            print('%0s, Score: %.5f' % (feature_names[i],v))

    features.drop(list(set(features.columns) - set(feature_names)), axis = 1)
    for new_col in list(set(feature_names) - set(features.columns)):
        features[new_col] = 0
    features = features[feature_names]
    features = features.replace(np.nan, 0)

    y_pred = loaded_model.predict(features)

    return y_pred

def build_features(flights_df):

    flights_df = flights_df.drop(['YEAR', 'CANCELLATION_REASON', 'DIVERTED', 'CANCELLED'], axis=1)

    #AIRLINE, TAIL_NUMBER, ORIGIN_AIRPORT, DESTINATION_AIRPORT
    arr_airline = flights_df.AIRLINE.unique()
    arr_tail_nbr = flights_df.TAIL_NUMBER.unique()
    arr_orig_airport = flights_df.ORIGIN_AIRPORT.unique()
    arr_dest_airport = flights_df.DESTINATION_AIRPORT.unique()
    map_values = dict(zip(arr_airline, range(len(arr_airline))))
    flights_df['AIRLINE'].replace(map_values, inplace=True)
    map_values = dict(zip(arr_tail_nbr, range(len(arr_tail_nbr))))
    flights_df['TAIL_NUMBER'].replace(map_values, inplace=True)
    map_values = dict(zip(arr_orig_airport, range(len(arr_orig_airport))))
    flights_df['ORIGIN_AIRPORT'].replace(map_values, inplace=True)
    map_values = dict(zip(arr_dest_airport, range(len(arr_dest_airport))))
    flights_df['DESTINATION_AIRPORT'].replace(map_values, inplace=True)

    #create labelled target variable for building the model
    flights_df['DELAY_FLAG'] = np.where(flights_df['ARRIVAL_DELAY'] > 0, 1, 0)
    flights_df = flights_df.drop(['ARRIVAL_DELAY'], axis=1)

    return flights_df

if __name__ == '__main0__':
    flights_df = load_data('flights.csv')
    print(flights_df.iloc[:10].to_string())
    print(flights_df.shape)

if __name__ == '__main__':
    flights_df = load_data('flights.csv')
    #first 100k records for building the model
    flights_df = build_features(flights_df.iloc[:100000])
    print(flights_df.iloc[:10].to_string())

    test_train(flights_df)

if __name__ == '__main2__':

    flights_df = load_data('flights.csv')
    #last 10k records for predicting the delays
    flights_df = build_features(flights_df.iloc[-10000:])
    results_df = flights_df.copy()
    #print(flights_df.columns.values)

    y_pred = predict_data(flights_df)
    results_df['Y_PRED'] = pd.Series(y_pred, index=results_df.index)
    results_df['_RESULT_'] = np.where(results_df['Y_PRED'] == results_df['DELAY_FLAG'], 1, 0)
    print("Predicted correct count :", results_df['_RESULT_'] .sum())
