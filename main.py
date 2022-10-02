
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

def load_airline():
    airline_data_df = pd.read_csv(DATA_FOLDER+'\\airlines.csv')
    airline_data_df = airline_data_df.sort_values(by=['AIRLINE'])
    print(airline_data_df)

def load_airports():
    airport_df = pd.read_csv(DATA_FOLDER + '\\airports.csv')
    airport_df = airport_df.sort_values(by=['AIRPORT'])
    df_first_10 = airport_df.iloc[:10]
    print(df_first_10)
    print(airport_df[['CITY', 'AIRPORT']])

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    load_airline()
    load_airports()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
