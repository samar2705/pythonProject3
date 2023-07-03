import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, binarize
from sklearn import metrics
import shap

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shared
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import pickle
import shap
from collections import Counter


def Know_Data(df):
    print('know The Data:\n')
    print('The DataFrame:\n', df.head(), '\n', ' DataFrame Info:\n',
          df.info(), '\n', ' DataFrame Describe:\n', df.describe(), '\n')

def data_preprocessing(df):
    # Check for missing values
    missing_values = df.isnull().sum()
    print(missing_values)

    # Drop rows with missing values
    df_cleaned = df.dropna()

    # Fill missing values with 0
    df.fillna(value=0, inplace=True)

    # Select numerical features
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns

    # Remove binary feature(s) from numerical features
    binary_features = [feat for feat in numerical_features if df[feat].nunique() == 2]
    numerical_features = list(set(numerical_features) - set(binary_features))

    # Create a MinMaxScaler object
    scaler = MinMaxScaler()

    # Normalize the numerical features
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Check for duplicate records
    duplicate_records = df.duplicated()
    # Count the number of duplicate records
    num_duplicates = duplicate_records.sum()
    # Remove duplicate records
    df = df.drop_duplicates()
    # Print the number of duplicate records removed
    print(f"Number of duplicate records removed: {num_duplicates}")

    return df
# Outliers Detection using the IQR method viewing in a boxplot
def remove_usersid(df):
    users = df['userid']
    df.drop(['userid','userid_0','userid_1','userid_2','touserid','fromuserid'], axis=1, inplace=True)
    return users


# def outliers_detection(df):
#     c = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, len(df.columns))]
#     fig = go.Figure()
#     fig.add_traces(data=[go.Box(
#         y=df.iloc[:, i],
#         marker_color=c[i],
#         name=df.columns[i])
#         for i in range(len(df.columns))])
#     removeUsers = shared.remove_usersid(df)
#
#     fig.update_layout(
#         title='Outliers Detection BoxPlot',)
#     fig.show()
# Outliers Detection using the IQR method viewing in a boxplot
def outliers_detection(df):
    df.drop('userid',axis=1,inplace=True)
    c = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, len(df.columns))]
    fig = go.Figure()
    fig.add_traces(data=[go.Box(
        y=df.iloc[:, i],
        marker_color=c[i],
        name=df.columns[i])
        for i in range(len(df.columns))
    ])
    fig.update_layout(
        title='Outliers Detection BoxPlot',
    )
    fig.show()
def find_skewed_bounaries(df, distance):

    data_trimmed_dropped = df.copy()
    data_trimmed_with_mean = df.copy()
    for variable in df.columns:
        IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
        if(IQR>3):
            lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
            upper_boundary = df[variable].quantile(0.75) + (IQR * distance)
            outliers_for_dropped = np.where(data_trimmed_dropped[variable] > upper_boundary, True,
                                            np.where(data_trimmed_dropped[variable] < lower_boundary, True, False))
            outliers_for_mean = np.where(data_trimmed_with_mean[variable] > upper_boundary, True,
                                         np.where(data_trimmed_with_mean[variable] < lower_boundary, True, False))
            data_trimmed_dropped = data_trimmed_dropped.loc[~(outliers_for_dropped)]
            variable_mean = df[variable].max()
            data_trimmed_with_mean[variable].loc[outliers_for_mean] = variable_mean

    return data_trimmed_dropped, data_trimmed_with_mean

#REMOVE BACK OFFICE (KPI = 0)
def remove_back_office(df,NumOfMonths):
    if(NumOfMonths=='7'):
        df = df[
            (df['kpi_m_1'] > 0) | (df['kpi_m_2'] > 0) | (df['kpi_m_3'] > 0) | (df['kpi_m_4'] > 0) | (
                        df['kpi_m_5'] > 0) | (
                    df['kpi_m_6'] > 0) | (df['kpi_m_7'] > 0)]
        df.reset_index(inplace=True, drop=True)
    else:
        df = df[
            (df['kpi_m_1'] > 0) | (df['kpi_m_2'] > 0) | (df['kpi_m_3'] > 0) | (df['kpi_m_4'] > 0)]
        df.reset_index(inplace=True, drop=True)
    return df


# def df_prepare_for_sub_group(df):
#     df_copy = df.copy()
#     scaler = MinMaxScaler()  # Create a MinMaxScaler object
#
#     # Loop through each column in the DataFrame
#     for col in df_copy.columns:
#         # Check if the column contains numeric data
#         if df_copy[col].dtype == 'int64' or df_copy[col].dtype == 'float64':
#             # Replace non-zero values with normalized values
#             non_zero_mask = df_copy[col] != 0  # Create a boolean mask for non-zero values
#             non_zero_values = df_copy[col][non_zero_mask]  # Select non-zero values
#             normalized_values = scaler.fit_transform(non_zero_values.values.reshape(-1, 1)).flatten()  # Normalize the non-zero values
#     return df_copy
def df_prepare_for_sub_group(df_copy):
    df_copy.fillna(0, inplace=True)
    for row in range(0, len(df_copy)):
        for column in df_copy.columns:
            if df_copy[column].iloc[row] != 0:
                df_copy[column].iloc[row] = 1
    return df_copy

# def df_prepare_for_sub_group(df):
#     df_copy = df.copy()
#     scaler = MinMaxScaler()  # Create a MinMaxScaler object
#
#     # Loop through each column in the DataFrame
#     for col in df_copy.columns:
#         # Check if the column contains numeric data
#         if df_copy[col].dtype == 'int64' or df_copy[col].dtype == 'float64':
#             # Replace non-zero values with normalized values
#             non_zero_mask = df_copy[col] != 0  # Create a boolean mask for non-zero values
#             non_zero_values = df_copy[col][non_zero_mask]  # Select non-zero values
#             if len(non_zero_values) > 0:
#                 normalized_values = scaler.fit_transform(non_zero_values.values.reshape(-1, 1)).flatten()  # Normalize the non-zero values
#                 df_copy.loc[non_zero_mask, col] = normalized_values  # Replace non-zero values with normalized values in the DataFrame
#
#     return df_copy
#
#

def create_sub_group(df):
    df_copy = df[['logins_m_1', 'logins_m_2', 'logins_m_3', 'logins_m_4', 'logins_m_5', 'logins_m_6', 'logins_m_7']]
    df_copy['sub_group'] = 0
    df_copy['group'] = 0

    ####################################################3
    df_copy = df_prepare_for_sub_group(df_copy)
    ####################################################3
    power = 0
    for column in df_copy.columns:
        df_copy[column] = df_copy[column].replace([1], pow(2, power))
        power += 1
    for row in range(0, len(df_copy)):
        result = 0
        for column in df_copy.columns:
            result += df_copy[column].iloc[row]
        if (result == 127) | (result == 126) | (result == 124) | (result == 120):
            df_copy['group'].iloc[row] = 'Stayed'
            df_copy['sub_group'].iloc[row] = '1'
        if (result == 112) or (result == 1) or (result == 3) or (result == 64) or (result == 96):
            df_copy['group'].iloc[row] = 'Noise'
            df_copy['sub_group'].iloc[row] = '2'

        if (result == 6) or (result == 12) or (result == 24) or (result == 48):
            df_copy['group'].iloc[row] = 'Left to drop'
            df_copy['sub_group'].iloc[row] = '3'
        if (result == 7) or (result == 14) or (result == 28) or (result == 56) or (result == 15) or (result == 30) or (
                result == 60) or (result == 31) or (result == 62) or (result == 63):
            df_copy['group'].iloc[row] = 'Left'
            if (result == 7):
                df_copy['sub_group'].iloc[row] = '4'
            if (result == 14):
                df_copy['sub_group'].iloc[row] = '5'
            if (result == 28):
                df_copy['sub_group'].iloc[row] = '6'
            if (result == 56):
                df_copy['sub_group'].iloc[row] = '7'
            if (result == 15):
                df_copy['sub_group'].iloc[row] = '8'
            if (result == 30):
                df_copy['sub_group'].iloc[row] = '9'
            if (result == 60):
                df_copy['sub_group'].iloc[row] = '10'
            if (result == 31):
                df_copy['sub_group'].iloc[row] = '11'
            if (result == 62):
                df_copy['sub_group'].iloc[row] = '12'
            if (result == 63):
                df_copy['sub_group'].iloc[row] = '13'

    print(df_copy)
    return df_copy[['sub_group','group']]

def feature_to_sub_group_5_8_9_11_12_13():
   return ['logins_m_2', 'logins_m_3', 'logins_m_4', 'kudos_t_m_2', 'kudos_t_m_3', 'kudos_t_m_4', 'kudos_f_m_2', 'kudos_f_m_3', 'kudos_f_m_4', 'kpi_m_2', 'kpi_m_3', 'kpi_m_4', 'lrn_m_2', 'lrn_m_3', 'lrn_m_4', 'lrn_t_m_2', 'lrn_t_m_3', 'lrn_t_m_4']

def feature_to_sub_group_6_10():
    return ['logins_m_3', 'logins_m_4', 'logins_m_5', 'kudos_t_m_3', 'kudos_t_m_4', 'kudos_t_m_5', 'kudos_f_m_3', 'kudos_f_m_4', 'kudos_f_m_5', 'kpi_m_3', 'kpi_m_4', 'kpi_m_5', 'lrn_m_3', 'lrn_m_4', 'lrn_m_5', 'lrn_t_m_3', 'lrn_t_m_4', 'lrn_t_m_5']

def feature_to_sub_group_1_7():
    return ['logins_m_4', 'logins_m_5', 'logins_m_6', 'kudos_t_m_4', 'kudos_t_m_5', 'kudos_t_m_6', 'kudos_f_m_4',
         'kudos_f_m_5', 'kudos_f_m_6', 'kpi_m_4', 'kpi_m_5', 'kpi_m_6', 'lrn_m_4', 'lrn_m_5', 'lrn_m_6', 'lrn_t_m_4',
         'lrn_t_m_5', 'lrn_t_m_6']

def feature_selection_And_prepare(df,NumOfMonths):
    if(NumOfMonths=='4'):
        for row in range(0, len(df)):
                if df['logins_m_4'].iloc[row] != 0:
                    df['logins_m_4'].iloc[row] = 0
                else:
                    df['logins_m_4'].iloc[row] = 1


        final_df = pd.DataFrame()
        final_df[['final_logins_m_1', 'final_logins_m_2', 'final_logins_m_3', 'final_kudos_t_m_1', 'final_kudos_t_m_2',
                  'final_kudos_t_m_3', 'final_kudos_f_m_1', 'final_kudos_f_m_2', 'final_kudos_f_m_3', 'final_kpi_m_1',
                  'final_kpi_m_2', 'final_kpi_m_3', 'final_lrn_m_1', 'final_lrn_m_2', 'final_lrn_m_3', 'final_lrn_t_m_1',
                  'final_lrn_t_m_2', 'final_lrn_t_m_3']] = df.drop(['logins_m_4', 'kudos_t_m_4', 'kudos_f_m_4', 'kpi_m_4',
                   'lrn_m_4', 'lrn_t_m_4'], axis=1)
        final_df['target'] = df['logins_m_4']
        final_df.reset_index(inplace=True, drop=True)
        print(final_df)

        # final_df = pd.DataFrame()
        # columns_to_drop = ['logins_m_4', 'kudos_t_m_4', 'kudos_f_m_4', 'kpi_m_4', 'lrn_m_4', 'lrn_t_m_4']
        # final_df = df.drop(columns_to_drop, axis=1)
        # final_df['target'] = df['logins_m_4']
        # final_df.reset_index(inplace=True, drop=True)

    else:
        #SUB GROUP 1
        columns_names_for_sub_1_7 = feature_to_sub_group_1_7()
        X_1 = pd.DataFrame()
        df_sub_1 = df[df['sub_group']=='1']
        df_sub_1.reset_index(inplace=True)
        df_sub_1 = df_sub_1.loc[:,columns_names_for_sub_1_7]
        X_1[['final_logins_m_1', 'final_logins_m_2', 'final_logins_m_3', 'final_kudos_t_m_1', 'final_kudos_t_m_2',
             'final_kudos_t_m_3', 'final_kudos_f_m_1', 'final_kudos_f_m_2', 'final_kudos_f_m_3', 'final_kpi_m_1',
             'final_kpi_m_2', 'final_kpi_m_3', 'final_lrn_m_1', 'final_lrn_m_2', 'final_lrn_m_3', 'final_lrn_t_m_1',
             'final_lrn_t_m_2', 'final_lrn_t_m_3']] = df_sub_1
        X_1['target'] = 0

        X_7 = pd.DataFrame()
        df_sub_7 = df[df['sub_group'] == '7']
        df_sub_7.reset_index(inplace=True)
        df_sub_7 = df_sub_7.loc[:,columns_names_for_sub_1_7]
        X_7[['final_logins_m_1', 'final_logins_m_2', 'final_logins_m_3', 'final_kudos_t_m_1', 'final_kudos_t_m_2',
             'final_kudos_t_m_3', 'final_kudos_f_m_1', 'final_kudos_f_m_2', 'final_kudos_f_m_3', 'final_kpi_m_1',
             'final_kpi_m_2', 'final_kpi_m_3', 'final_lrn_m_1', 'final_lrn_m_2', 'final_lrn_m_3', 'final_lrn_t_m_1',
             'final_lrn_t_m_2', 'final_lrn_t_m_3']] = df_sub_7
        X_7['target'] = 1
        ###########################################################################################
        #SUB GROUP 4
        X_4 = pd.DataFrame()
        df_sub_4 = df[df['sub_group']=='4']
        df_sub_4.reset_index(inplace=True)
        df_sub_4 = df_sub_4.loc[:,['logins_m_1', 'logins_m_2', 'logins_m_3', 'kudos_t_m_1', 'kudos_t_m_2', 'kudos_t_m_3', 'kudos_f_m_1', 'kudos_f_m_2', 'kudos_f_m_3', 'kpi_m_1', 'kpi_m_2', 'kpi_m_3', 'lrn_m_1', 'lrn_m_2', 'lrn_m_3', 'lrn_t_m_1', 'lrn_t_m_2', 'lrn_t_m_3']]
        X_4[['final_logins_m_1', 'final_logins_m_2', 'final_logins_m_3', 'final_kudos_t_m_1', 'final_kudos_t_m_2',
             'final_kudos_t_m_3', 'final_kudos_f_m_1', 'final_kudos_f_m_2', 'final_kudos_f_m_3', 'final_kpi_m_1',
             'final_kpi_m_2', 'final_kpi_m_3', 'final_lrn_m_1', 'final_lrn_m_2', 'final_lrn_m_3', 'final_lrn_t_m_1',
             'final_lrn_t_m_2', 'final_lrn_t_m_3']] = df_sub_4
        X_4['target'] = 1
        ###########################################################################################
        # SUB GROUP 5 + 8 + 9 + 11 + 12 + 13
        culomns_names = feature_to_sub_group_5_8_9_11_12_13()
        X_5 = pd.DataFrame()
        X_8 = pd.DataFrame()
        X_9 = pd.DataFrame()
        X_11 = pd.DataFrame()
        X_12 = pd.DataFrame()
        X_13 = pd.DataFrame()

        df_sub_5 = df[df['sub_group'] == '5']
        df_sub_5.reset_index(inplace=True)
        df_sub_5 = df_sub_5.loc[:,culomns_names]
        df_sub_8 = df[df['sub_group'] == '8']
        df_sub_8.reset_index(inplace=True)
        df_sub_8 = df_sub_8.loc[:,culomns_names]
        df_sub_9 = df[df['sub_group'] == '9']
        df_sub_9.reset_index(inplace=True)
        df_sub_9 = df_sub_9.loc[:,culomns_names]
        df_sub_11 = df[df['sub_group'] == '11']
        df_sub_11.reset_index(inplace=True)
        df_sub_11 = df_sub_11.loc[:,culomns_names]
        df_sub_12 = df[df['sub_group'] == '12']
        df_sub_12.reset_index(inplace=True)
        df_sub_12 = df_sub_12.loc[:,culomns_names]
        df_sub_13 = df[df['sub_group'] == '13']
        df_sub_13.reset_index(inplace=True)
        df_sub_13 = df_sub_13.loc[:,culomns_names]
        X_5[['final_logins_m_1', 'final_logins_m_2', 'final_logins_m_3', 'final_kudos_t_m_1', 'final_kudos_t_m_2',
             'final_kudos_t_m_3', 'final_kudos_f_m_1', 'final_kudos_f_m_2', 'final_kudos_f_m_3', 'final_kpi_m_1',
             'final_kpi_m_2', 'final_kpi_m_3', 'final_lrn_m_1', 'final_lrn_m_2', 'final_lrn_m_3', 'final_lrn_t_m_1',
             'final_lrn_t_m_2', 'final_lrn_t_m_3']] = df_sub_5
        X_8[['final_logins_m_1', 'final_logins_m_2', 'final_logins_m_3', 'final_kudos_t_m_1', 'final_kudos_t_m_2',
             'final_kudos_t_m_3', 'final_kudos_f_m_1', 'final_kudos_f_m_2', 'final_kudos_f_m_3', 'final_kpi_m_1',
             'final_kpi_m_2', 'final_kpi_m_3', 'final_lrn_m_1', 'final_lrn_m_2', 'final_lrn_m_3', 'final_lrn_t_m_1',
             'final_lrn_t_m_2', 'final_lrn_t_m_3']] = df_sub_8
        X_9[['final_logins_m_1', 'final_logins_m_2', 'final_logins_m_3', 'final_kudos_t_m_1', 'final_kudos_t_m_2',
             'final_kudos_t_m_3', 'final_kudos_f_m_1', 'final_kudos_f_m_2', 'final_kudos_f_m_3', 'final_kpi_m_1',
             'final_kpi_m_2', 'final_kpi_m_3', 'final_lrn_m_1', 'final_lrn_m_2', 'final_lrn_m_3', 'final_lrn_t_m_1',
             'final_lrn_t_m_2', 'final_lrn_t_m_3']] = df_sub_9
        X_11[['final_logins_m_1', 'final_logins_m_2', 'final_logins_m_3', 'final_kudos_t_m_1', 'final_kudos_t_m_2',
             'final_kudos_t_m_3', 'final_kudos_f_m_1', 'final_kudos_f_m_2', 'final_kudos_f_m_3', 'final_kpi_m_1',
             'final_kpi_m_2', 'final_kpi_m_3', 'final_lrn_m_1', 'final_lrn_m_2', 'final_lrn_m_3', 'final_lrn_t_m_1',
             'final_lrn_t_m_2', 'final_lrn_t_m_3']] = df_sub_11
        X_12[['final_logins_m_1', 'final_logins_m_2', 'final_logins_m_3', 'final_kudos_t_m_1', 'final_kudos_t_m_2',
             'final_kudos_t_m_3', 'final_kudos_f_m_1', 'final_kudos_f_m_2', 'final_kudos_f_m_3', 'final_kpi_m_1',
             'final_kpi_m_2', 'final_kpi_m_3', 'final_lrn_m_1', 'final_lrn_m_2', 'final_lrn_m_3', 'final_lrn_t_m_1',
             'final_lrn_t_m_2', 'final_lrn_t_m_3']] = df_sub_12
        X_13[['final_logins_m_1', 'final_logins_m_2', 'final_logins_m_3', 'final_kudos_t_m_1', 'final_kudos_t_m_2',
             'final_kudos_t_m_3', 'final_kudos_f_m_1', 'final_kudos_f_m_2', 'final_kudos_f_m_3', 'final_kpi_m_1',
             'final_kpi_m_2', 'final_kpi_m_3', 'final_lrn_m_1', 'final_lrn_m_2', 'final_lrn_m_3', 'final_lrn_t_m_1',
             'final_lrn_t_m_2', 'final_lrn_t_m_3']] = df_sub_13
        X_5['target'] = 1
        X_8['target'] = 1
        X_9['target'] = 0
        X_11['target'] = 0
        X_12['target'] = 0
        X_13['target'] = 0

        ###########################################################################################
        # SUB GROUP 6 + 1 0
        columns_names_for_sub_6_10 = feature_to_sub_group_6_10()
        X_6 = pd.DataFrame()
        X_10 = pd.DataFrame()
        df_sub_6 = df[df['sub_group'] == '6']
        df_sub_6.reset_index(inplace=True)
        df_sub_6 = df_sub_6.loc[:,columns_names_for_sub_6_10]
        df_sub_10 = df[df['sub_group'] == '10']
        df_sub_10.reset_index(inplace=True)
        df_sub_10 = df_sub_10.loc[:,columns_names_for_sub_6_10]
        X_6[['final_logins_m_1', 'final_logins_m_2', 'final_logins_m_3', 'final_kudos_t_m_1', 'final_kudos_t_m_2',
             'final_kudos_t_m_3', 'final_kudos_f_m_1', 'final_kudos_f_m_2', 'final_kudos_f_m_3', 'final_kpi_m_1',
             'final_kpi_m_2', 'final_kpi_m_3', 'final_lrn_m_1', 'final_lrn_m_2', 'final_lrn_m_3', 'final_lrn_t_m_1',
             'final_lrn_t_m_2', 'final_lrn_t_m_3']] = df_sub_6
        X_10[['final_logins_m_1', 'final_logins_m_2', 'final_logins_m_3', 'final_kudos_t_m_1', 'final_kudos_t_m_2',
             'final_kudos_t_m_3', 'final_kudos_f_m_1', 'final_kudos_f_m_2', 'final_kudos_f_m_3', 'final_kpi_m_1',
             'final_kpi_m_2', 'final_kpi_m_3', 'final_lrn_m_1', 'final_lrn_m_2', 'final_lrn_m_3', 'final_lrn_t_m_1',
             'final_lrn_t_m_2', 'final_lrn_t_m_3']] = df_sub_10
        X_6['target'] = 1
        X_10['target'] = 0

        ###########################################################################################
        ###########################################################################################
        final_df = pd.concat([X_1,X_4,X_5,X_6,X_7,X_8,X_9,X_10,X_11,X_12,X_13])
        final_df.reset_index(inplace = True,drop=True)
    return final_df


#Scaling helps us understanding the relative contribution of each feature to the model,it make all features in the same scale.
def data_scaling(df):
    # Scaling - MinMax, This technique re-scales a feature or observation value with distribution value between 0 and 1. :
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df),columns=df.columns)
    return df




# because our data is Imbalanced(A classification data set with skewed class proportions is called imbalanced),we use the smote algorithm that we here oversmaple the minory class)
def smote_prepare_and_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
    smote = SMOTE(random_state=1)
    counter = Counter(y_train)
    X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, y_train)
    counter = Counter(Y_train_resampled)
    return X_train_resampled, X_test, Y_train_resampled, y_test

# # Separate features and target variable
# X = df.drop('target', axis=1)
# y = df['target']
#
# # Apply SMOTE for oversampling
# smote = SMOTE()
# X_resampled, y_resampled = smote.fit_resample(X, y)

# run  Logistic regression model with using smote
def Logistic_regression(X,y):
    X_train_logr, X_test_logr, y_train_logr, y_test_logr = smote_prepare_and_split(X, y)
    logmodel = LogisticRegression()
    logmodel.fit(X_train_logr, y_train_logr)
    pickle.dump(logmodel, open('Logistic_regression.pkl', 'wb'))
    probability_ = logmodel.predict_proba(X_test_logr)
    log_pred = logmodel.predict(X_test_logr)
    print('Logistic_regression evaluate results:')
    AUC_ROC_evaluate(probability_,log_pred,y_test_logr)
    explainer = shap.Explainer(logmodel, X_train_logr)  # Assuming X_train_logr is the training data used for the model
    shap_values = explainer.shap_values(X_test_logr)
    X_test_logr_resetd_index = X_test_logr.reset_index()
    row_index = 152  # Index of the specific row you want to explain
    row_to_explain = X_test_logr.iloc[row_index]  # Assuming X_test_logr is your test data
    f = shap.force_plot(explainer.expected_value, shap_values[row_index], row_to_explain)
    shap.save_html("Logistic_regression shap results.html", f)
    print_shap_values(X_test_logr.columns, shap_values[row_index])
    shap.summary_plot(shap_values, X_test_logr)

def print_shap_values(features, shap_values):
    print("Feature    SHAP Value")
    print("---------------------")
    for feature, shap_value in zip(features, shap_values):
        print(f"{feature:<10} {shap_value:.2f}")

#def show_importance(model,X):
    #importances = model.feature_importances_
    #names=model.rf.feature_names_in_
    #importances_df = pd.DataFrame({"feature_names": importances,
                                  # "importances": names})
    #print('The Feature Importance \n')
    #print(importances)
    #g = sns.barplot(x=importances_df["feature_names"],
       #             y=importances_df["importances"])
    #g.set_title("Feature importances", fontsize=14);
    #fig, ax = plt.subplots()
    #model_importances.plot.bar(ax=ax)
    #ax.set_title("Feature importances")
    #ax.set_ylabel("To Be continue")
    #fig.tight_layout()
# run  random Forest model with using smote
def random_Forest(X,y):
    X_trainRANF, X_testRANF, y_trainRANF, y_testRANF = smote_prepare_and_split(X, y)
    rforest = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
    rforest.fit(X_trainRANF, y_trainRANF)
    #Show Importance Function
    #show_importance(rforest,X_trainRANF)
    pickle.dump(rforest, open('random_Forest.pkl', 'wb'))
    rforest_proba = rforest.predict_proba(X_testRANF)
    rforest_pred = rforest.predict(X_testRANF)
    print('random_Forest evaluate results:')
    AUC_ROC_evaluate(rforest_proba,rforest_pred,y_testRANF)
    explainer = shap.Explainer(rforest, X_testRANF)  # Assuming X_train_logr is the training data used for the model

    rows_list_to_shap = [487,489,490,550,1,473,480,484] # Index of the specific row you want to explain
    for i in rows_list_to_shap:
      row_to_explain = X_testRANF.iloc[i]  # Assuming X_test is your test data
      shap_values = explainer.shap_values(row_to_explain, check_additivity=False)
      print_shap_values(X_testRANF.columns, shap_values[1])
      print(shap_values)
      shap.initjs()
      f = shap.force_plot(explainer.expected_value[1], shap_values[1], row_to_explain)
      shap.save_html(f"random_Forest shap results {i}.html", f)
    shap_values_summary = explainer.shap_values(X_testRANF, check_additivity=False)
    shap.summary_plot(shap_values_summary, X_testRANF)


def evaluate_model_metrics(model, X_test, y_test):
    # Make predictions on the testing data using the trained model
    y_pred = model.predict(X_test)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    # Calculate recall (sensitivity)
    recall = recall_score(y_test, y_pred)
    print("Recall:", recall)
    # Calculate AUC-ROC
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc_roc = roc_auc_score(y_test, y_pred_prob)
    print("AUC-ROC:", auc_roc)
    # Calculate balanced accuracy
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print("Balanced Accuracy:", balanced_acc)
    return accuracy, recall, auc_roc, balanced_acc


#evaluate our models,display Auc,classification_report with diffrent ThreShold
def AUC_ROC_evaluate(probability, prediction, y_test):
    y_test_prob_0 = pd.Series(probability[:, 0], name='probability_0', index=y_test.index)
    y_test_prob_1 = pd.Series(probability[:, 1], name='probability_1', index=y_test.index)
    y_test_pred = pd.Series(prediction, name='prediction', index=y_test.index)
    uni_test_results = pd.DataFrame(
        data={'probability_0': y_test_prob_0, 'probability_1': y_test_prob_1, 'prediction': y_test_pred,
              'actual': y_test})
    y_pred_prob = uni_test_results.probability_1
    y_test_pred = uni_test_results.prediction
    y_pred_03 = binarize(X=[y_pred_prob], threshold=0.3)[0]
    y_pred_03 = pd.Series(y_pred_03)
    y_pred_08 = binarize(X=[y_pred_prob], threshold=0.8)[0]
    y_pred_08 = pd.Series(y_pred_08)
    print(classification_report(y_test, y_test_pred))
    print('ThreShold = 0.3')
    print(classification_report(y_test, y_pred_03))
    print('ThreShold = 0.8')
    print(classification_report(y_test, y_pred_08))
    AUC = metrics.roc_auc_score(y_test, y_pred_prob)
    print('AUC : ', AUC)

def trend(df):
    df['Trend_logins_1']=df['final_logins_m_2']-df['final_logins_m_1']
    df['Trend_logins_2'] = df['final_logins_m_3'] - df['final_logins_m_2']
    df['Trend_kudos_t_m_1'] = df['final_kudos_t_m_2'] - df['final_kudos_t_m_1']
    df['Trend_kudos_t_m_2'] = df['final_kudos_t_m_3'] - df['final_kudos_t_m_2']
    df['Trend_kudos_f_m_1'] = df['final_kudos_f_m_2'] - df['final_kudos_f_m_1']
    df['Trend_kudos_f_m_2'] = df['final_kudos_f_m_3'] - df['final_kudos_f_m_2']
    df['Trend_kpi_m_1'] = df['final_kpi_m_2'] - df['final_kpi_m_1']
    df['Trend_kpi_m_2'] = df['final_kpi_m_3'] - df['final_kpi_m_2']
    df['Trend_lrn_m_1'] = df['final_lrn_m_2'] - df['final_lrn_m_1']
    df['Trend_lrn_m_2'] = df['final_lrn_m_3'] - df['final_lrn_m_2']
    df['Trend_lrn_t_m_1'] = df['final_lrn_t_m_2'] - df['final_lrn_t_m_1']
    df['Trend_lrn_t_m_2'] = df['final_lrn_t_m_3'] - df['final_lrn_t_m_2']

    # Train a model
    X = df.drop(['Trend_logins_1', 'Trend_logins_2', 'Trend_kudos_t_m_1', 'Trend_kudos_t_m_2', 'Trend_kudos_f_m_1',
                 'Trend_kudos_f_m_2', 'Trend_kpi_m_1', 'Trend_kpi_m_2', 'Trend_lrn_m_1', 'Trend_lrn_m_2',
                 'Trend_lrn_t_m_1', 'Trend_lrn_t_m_2', 'target'], axis=1)  # Remove the target column
    y = df['target']  # Replace 'target' with your actual target variable name
    trendmodel = LogisticRegression()  # Replace with your desired model
    trendmodel.fit(X, y)

    # Compute SHAP values
    explainer = shap.Explainer(trendmodel, X)
    shap_values = explainer.shap_values(X)

    # Visualize SHAP summary plot
    shap.summary_plot(shap_values, X, plot_type="bar")
    plt.show()

    return df

    # # Train a model
    # X = df.drop(['Trend_logins_1', 'Trend_logins_2', 'Trend_kudos_t_m_1', 'Trend_kudos_t_m_2', 'Trend_kudos_f_m_1',
    #                  'Trend_kudos_f_m_2', 'Trend_kpi_m_1', 'Trend_kpi_m_2', 'Trend_lrn_m_1', 'Trend_lrn_m_2',
    #                  'Trend_lrn_t_m_1', 'Trend_lrn_t_m_2'], axis=1)
    # y = df['target']  # Replace 'target_variable' with your actual target variable name
    # trendmodel = LogisticRegression()  # Replace with your desired model
    # trendmodel.fit(X, y)
    # # Compute SHAP values
    # explainer = shap.Explainer(trendmodel,X)
    # shap_values = explainer.shap_values(X)
    # print_shap_values(X.columns, shap_values)
    #
    # # Visualize SHAP summary plot
    # shap.summary_plot(shap_values, X)
    # plt.show()
    #
    # return df


# def comparing(df):
#     # Access and manipulate the data using column names
#     final_logins_m_3 = df['final_logins_m_3']
#     final_kpi_m_2 = df['final_kpi_m_2']
#     final_kpi_m_3 = df['final_kpi_m_3']
#     final_lrn_m_1 = df['final_lrn_m_1']
#     final_lrn_m_3 = df['final_lrn_m_3']
#     final_lrn_t_m_2 = df['final_lrn_t_m_2']
#     final_lrn_t_m_3 = df['final_lrn_t_m_3']
#     Trend_logins_1 = df['Trend_logins_1']
#     Trend_logins_2 = df['Trend_logins_2']
#     Trend_kpi_m_2 = df['Trend_kpi_m_2']
#     Trend_lrn_m_2 = df['Trend_lrn_m_2']
#     Trend_lrn_t_m_1 = df['Trend_lrn_t_m_1']
#     target = df['target']
#
#     # Perform comparisons between columns
#     comparison_result = final_logins_m_3 > final_kpi_m_2
#
#     # Print the comparison result
#     print(comparison_result)
#     return comparison_result


