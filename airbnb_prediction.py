# Predict the airbnb price using New York City Airbnb Open Data
# The dataset is downloaded from Kaggle (https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) and
# saved as AB_NYC_2019.csv

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm

def mutual_info(series):
    return mutual_info_score(series, df_train_full.price)


if __name__ == "__main__":
    df = pd.read_csv('AB_NYC_2019.csv')

    # Replace spaces in feature names with '_'
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    print(df.columns)

    # Get features that are object (categorical) and replace ' ' (in their values) with '_'
    strings = list(df.dtypes[df.dtypes=='object'].index)
    print(strings)
    for col in strings:
        df[col] = df[col].str.lower().str.replace(' ', '_')

    # EDA
    for col in strings:
        print(col)
        print(df[col].unique())
        print(df[col].nunique())
        print()

    # Distribution of price
    sns.histplot(df['price'])
    plt.savefig('hist_price1.png')
    plt.close()

    sns.histplot(df['price'][df['price'] <= 1000], bins=50)
    plt.savefig('hist_price2.png')
    plt.close()

    # Count number of null values
    print('Count null: \n', df.isnull().sum())

    # Numerical and categorical features
    numerical_features = ['latitude',
                          'longitude',
                          'minimum_nights',
                          'number_of_reviews',
                          'reviews_per_month',
                          'calculated_host_listings_count',
                          'availability_365']
    categorical_features = ['neighbourhood_group', 'room_type']

    # Fill the missing values with 0
    df[numerical_features + categorical_features] = df[numerical_features + categorical_features].fillna(0)

    #  The most frequent observation (mode) for the column 'neighbourhood_group'
    # print('Mode: \n', df['neighbourhood_group'].mode())

    # Split dataset into subsets (60%, 20%. 20%)
    df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=42)

    # Reset index
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # Target
    y_train_full = np.log1p(df_train_full.price.values)
    y_train = np.log1p(df_train.price.values)
    y_val = np.log1p(df_val.price.values)
    y_test = np.log1p(df_test.price.values)

    # Delete "price" from subsets
    del df_train['price']
    del df_val['price']
    del df_test['price']

    # Feature importance analysis
    ## Mutual information: (for categorical variables)
    mutual_info = df_train_full[categorical_features].apply(mutual_info)
    print('Mutual info: \n', mutual_info)

    ## Correlation (for numerical variables)
    correlation = df_train_full[numerical_features].corrwith(df_train_full.price).abs().sort_values(ascending=True)
    print('Correlation:\n', correlation)

    # availability_365
    # longitude
    # neighbourhood_group    0.105645
    # room_type

    # One-hot encoding
    dv = DictVectorizer(sparse=False)
    train_dict = df_train[numerical_features + categorical_features].to_dict(orient = 'records')
    X_train = dv.fit_transform(train_dict)
    val_dict = df_val[numerical_features + categorical_features].to_dict(orient = 'records')
    X_val = dv.transform(val_dict)
    test_dict = df_test[numerical_features + categorical_features].to_dict(orient = 'records')
    X_test = dv.transform(test_dict)

    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print('Model intercept: ', model.intercept_.round(3))
    print('Model coefficients: ', model.coef_.round(3))
    print (dict(zip(dv.get_feature_names_out(), model.coef_.round(3))))
    pred_y_train = model.predict(X_train)
    rmse_train = mean_squared_error(pred_y_train, y_train, squared=False)
    print('rmse_train: ', rmse_train)
    ## Evaluate the model
    pred_y_val = model.predict(X_val)
    rmse_val = mean_squared_error(pred_y_val, y_val, squared=False)
    print('rmse_val: ', rmse_val)

    # Train regularized regression with r = 0.01
    model = Ridge(alpha=0.1)
    model.fit(X_train, y_train)
    pred_y_train = model.predict(X_train)
    rmse_train = mean_squared_error(pred_y_train, y_train, squared=False)
    print('rmse_train - regularized: ', rmse_train)
    pred_y_val = model.predict(X_val)
    rmse_val = mean_squared_error(pred_y_val, y_val, squared=False)
    print('rmse_val - regularized: ', rmse_val)

    ## Tuning the model
    for r in tqdm([0.0, 0.0001, 0.001, 0.01, 1, 10]):
        model = Ridge(alpha=r)
        model.fit(X_train, y_train)
        pred_y_val = model.predict(X_val)
        rmse_val = mean_squared_error(pred_y_val, y_val, squared=False)
        print('Regularized model: r = %f, rmse_val = %f' %(r, rmse_val))

    # The model achieve the best performance with linear regression
    # Train the model on the full train set
    full_train_dict = df_train_full[numerical_features+categorical_features].to_dict(orient='records')
    dv = DictVectorizer()
    X_train_full = dv.fit_transform(full_train_dict)
    X_test = dv.transform(test_dict)

    model = LinearRegression()
    model.fit(X_train_full, y_train_full)
    pred_y_test = model.predict(X_test)
    rmse_test = mean_squared_error(pred_y_test, y_test, squared=False)
    print('rmse_test: ', rmse_test)

    # Plot ground-truth and predicted values
    sns.histplot(y_test, color='blue')
    sns.histplot(pred_y_test, color='red')
    plt.legend(labels = ["ground-truth price", "predicted price"])
    plt.savefig('plot_groundtruth_predicted')
    plt.close()
