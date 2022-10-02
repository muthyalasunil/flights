
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import datetime
import pandas as pd

DATA_FOLDER = 'data'

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    #print(f'Hi, {name} today is ')  # Press Ctrl+F8 to toggle the breakpoint.
    print('Hi %s, today is: %s' % (name, datetime.datetime.today().strftime("%Y / %m /%d")))

def calculate(number1,number2):
    product=number1*number2
    print('the product of %s and %s is %s' % (number1, number2, product))

def load_data(filename):
    airline_data_df = pd.read_csv(DATA_FOLDER+'\\'+filename)
    return airline_data_df


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    airline_df = load_data('airlines.csv')
    airline_df = airline_df.sort_values(by=['AIRLINE'])
    print(airline_df)

    airport_df = load_data('airports.csv')
    airport_df = airport_df.sort_values(by=['AIRPORT'])
    df_first_10 = airport_df.iloc[:10]
    print(df_first_10[['CITY', 'AIRPORT']])

    flights_df = load_data('flights.csv')
    #flights_df = flights_df.iloc[:10]
    print(flights_df.shape)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
