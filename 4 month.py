import clf as clf
import pandas as pd
import pickle
import shared
from pandas import read_csv

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split



#################################################################################
# DATA READING & Preprocessing

data = pd.read_csv(r"C:\Users\samar\OneDrive\Desktop\Final project\logins_kodus_kpi_lrn_left_synchronyfinancial_used_jan_4m.csv")
shared.Know_Data(data)
data = shared.data_preprocessing(data)
#################################################################################
# REMOVE BACK OFFICE (KPI = 0)
data = shared.remove_back_office(data, '4')
#################################################################################
# OUTLIERS DETECTION AND HANDLING
print(data.shape)
data_trimmed_dropped, data_trimmed_with_mean = shared.find_skewed_bounaries(data, 6)
shared.outliers_detection(data)
shared.outliers_detection(data_trimmed_with_mean)
shared.outliers_detection(data_trimmed_dropped)
print(data_trimmed_dropped.shape)
#################################################################################

# feature_selection_And_prepare&&Scaling
#group_divition = shared.create_sub_group(data_trimmed_dropped)
#data_trimmed_dropped['sub_group'] = group_divition['sub_group']
#data_trimmed_dropped['group'] = group_divition['group']
#print(data_trimmed_dropped)
final_df = shared.feature_selection_And_prepare(data_trimmed_dropped, '4')
final_df = shared.data_scaling(final_df)
print(final_df)
##################################################################################
# Run 2 models with evaluation
X = final_df.drop('target', axis=1)
y = final_df['target']
shared.Logistic_regression(X, y)
shared.random_Forest(X, y)

#Trend
final_df = shared.trend(final_df)
X = final_df.drop('target', axis=1)
y = final_df['target']
shared.Logistic_regression(X, y)
shared.random_Forest(X, y)


# Making predictions and evaluating the models
X_train_log, X_test_log, Y_train_log, y_test_log = shared.smote_prepare_and_split(X, y)
X_trainRANF, X_testRANF, y_trainRANF, y_testRANF = shared.smote_prepare_and_split(X, y)

load_model_for_logistic = pickle.load(open('Logistic_regression.pkl', 'rb'))
load_model_for_rf = pickle.load(open('random_Forest.pkl', 'rb'))

# use the loaded model to make predictions
y_pred_prob_log = load_model_for_logistic.predict_proba(X_test_log)
y_pred_log = load_model_for_logistic.predict(X_test_log)

y_pred_prob_rf = load_model_for_rf.predict_proba(X_testRANF)
y_pred_rf = load_model_for_rf.predict(X_testRANF)

shared.AUC_ROC_evaluate(y_pred_prob_log, y_pred_log, y_test_log)
shared.AUC_ROC_evaluate(y_pred_prob_rf, y_pred_rf, y_testRANF)



