# import libraries
import utils
import pandas as pd
import projectpro


# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# read data
data = pd.read_csv('insurance.csv')
print("Data read!")


# split data
X_train, X_test, y_train, y_test = utils.split_data(data, 'charges', 0.33, 42)
projectpro.checkpoint('1e808c')
print("Data split done!")


# process and train linear regression
X_train, X_test, y_train, y_test, y_train_t, y_test_t, pt= utils.process_data_for_LR(X_train, X_test, y_train, y_test)
print("Data processing for Linear Regression done!")

# Train LR
y_pred_train, y_pred_test, lr = utils.train_and_evaluate_LR(X_train, X_test, y_train, y_test, y_train_t, y_test_t, pt)


# Data Preparation for XGBoost
print("Preparing data for XGBoost modelling!")
X_train, X_test, y_train, y_test = utils.split_data(data, 'charges', 0.33, 42)

X_train, X_test, ohe = utils.process_data_for_xgboost(X_train, X_test)
print("Data one hot encoded for xgboost")

# train xgboost and evaluate xgboost
y_pred_train_xgb, y_pred_test_xgb, xgb_bs_cv = utils.train_and_evaluate_xgboost(X_train, X_test, y_train, y_test)
projectpro.checkpoint('1e808c')


