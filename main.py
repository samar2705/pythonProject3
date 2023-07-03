import shap
import shared
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Read the CSV file into a pandas DataFrame
warnings.filterwarnings("ignore")
data = pd.read_csv(r"C:\Users\samar\OneDrive\Desktop\Final project\sampling.csv")

# Print initial information about the data
shared.Know_Data(data)

# Preprocess the data
data = shared.data_preprocessing(data)
data = shared.remove_back_office(data, NumOfMonths='7')

# Outliers detection
# data = shared.outliers_detection(data)
# Users = shared.remove_usersid(data)

# OUTLIERS DETECTION AND HANDLING
print(data)
data_trimmed_dropped, data_trimmed_with_mean = shared.find_skewed_bounaries(data, 6)
shared.outliers_detection(data)
shared.outliers_detection(data_trimmed_with_mean)
shared.outliers_detection(data_trimmed_dropped)
print(data_trimmed_dropped.shape)
#print(data_trimmed_dropped)

# feature_selection_And_prepare&&Scaling
group_divition = shared.create_sub_group(data_trimmed_dropped)
data_trimmed_dropped['sub_group'] = group_divition['sub_group']
data_trimmed_dropped['group'] = group_divition['group']
print(data_trimmed_dropped)
final_df = shared.feature_selection_And_prepare(data_trimmed_dropped, '7')
final_df = shared.data_scaling(final_df)
print(final_df)

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

print(final_df.shape)

#final_df.to_csv('final_df.csv', index=False)

# data1 = pd.read_csv(r"C:\Users\samar\OneDrive\Documents\data_export.csv")
# compare_df = shared.comparing(data1)
#
