# For load and manipulating data.
import pandas as pd
import numpy as np

# For visualization purposes.
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import folium
from folium.plugins import HeatMap, MarkerCluster

# For KDE and regression models.
import mlpack
from mlpack import preprocess_split
from mlpack import LinearRegression
from mlpack import lars
from mlpack import bayesian_linear_regression
from mlpack import linear_svm

# For cross validation.
from sklearn.model_selection import KFold

# For file and os systems
import os
import sys

""" Model Evaluation
"""
def r2score(y_true, y_preds):
    corr_matrix = np.corrcoef(y_true, y_preds)
    corr = corr_matrix[0,1]
    R2 = corr**2
    return R2

def modelEval(ytest, yPreds):
    print("\n---- Evaluation Metrics ----")
    print(f"Mean Absoulte Error: {np.mean(np.abs(yPreds - ytest)):.2f}")
    print(f"Mean Squared Error: {np.mean(np.power(yPreds - ytest, 2)):.2f}")
    print(f"Root Mean Squared Error: {np.sqrt(np.mean(np.power(yPreds - ytest, 2))):.2f}")
    print(f"R2 score: {r2score(ytest, yPreds):.2f}")

def get_evaluation_scores(y_test, y_pred):
    mae = np.mean(np.abs(y_test - y_pred))
    mse = np.mean(np.power(y_test - y_pred, 2))
    rmse = np.sqrt(np.mean(np.power(y_test - y_pred, 2)))
    r2_score = r2score(y_test, y_pred)
    return [mae, mse, rmse, r2_score]

""" Main Function
"""
if __name__ == "__main__":
    
    # read dataset
    input = './input/osc_cc_runtime_dataset_r0_nmax100.csv'
    df = pd.read_csv(input, header=None)
    df.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'y']
    
    # extract features and labels for training
    features = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    X = df[features]
    y = df['y']

    # -----------------------------------------------------
    # Regression Models
    # -----------------------------------------------------
    lr_model = LinearRegression()
    rr_model = LinearRegression(lambda_=0.5)
    
    num_models = 4
    k = 5
    mae_arr = []
    mse_arr = []
    rmse_arr = []
    r2_score_arr = []
    for i in range(num_models):
        mae_arr.append([])
        mse_arr.append([])
        rmse_arr.append([])
        r2_score_arr.append([])
    
    kf = KFold(n_splits=k, random_state = None)
    for train_index , test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index,:],X.iloc[test_index,:]
        y_train, y_test = np.array(y[train_index]) , np.array(y[test_index])

        # linear regression
        output_lr = lr_model.fit(training=X_train, training_responses=y_train)
        predic_lr = lr_model.predict(test=X_test)
        score_lr  = get_evaluation_scores(y_test, predic_lr)

        # ridge regression
        output_rr = rr_model.fit(training=X_train, training_responses=y_train)
        predic_rr = rr_model.predict(test=X_test)
        score_rr  = get_evaluation_scores(y_test, predic_rr)

        # bayesian_linear_regression
        output_blr = bayesian_linear_regression(input_=X_train, responses=y_train)
        blr_model = output_blr['output_model']
        blr_predict = bayesian_linear_regression(input_model=blr_model, test=X_test)
        predic_blr = blr_predict['predictions'].reshape(-1, 1).squeeze()
        score_blr  = get_evaluation_scores(y_test, predic_blr)

        # lars
        X_tran_transposed = X_train.T # a bug from mlpack
        X_test_transposed = X_test.T
        output_lars = lars(input_=X_tran_transposed, responses=y_train, lambda1=0.4, lambda2=0.1)
        lars_model = output_lars['output_model']
        lars_predict = lars(input_model=lars_model, test=X_test_transposed)
        predic_lars = lars_predict['output_predictions'].reshape(-1, 1).squeeze()
        score_lars  = get_evaluation_scores(y_test, predic_lars)

        # svm
        # output_svm = linear_svm(training=X_train, labels=y_train, lambda_=0.5, delta=0.5, num_classes=6)
        # svm_model = output_svm['output_model']
        # svm_predict = linear_svm(input_model=svm_model, test=X_test)
        # predic_svm = svm_predict['predictions'].reshape(-1, 1).squeeze()
        # score_svm  = get_evaluation_scores(y_test, predic_svm)

        # record the scores
        mae_arr[0].append(score_lr[0])
        mae_arr[1].append(score_rr[0])
        mae_arr[2].append(score_blr[0])
        mae_arr[3].append(score_lars[0])

        mse_arr[0].append(score_lr[1])
        mse_arr[1].append(score_rr[1])
        mse_arr[2].append(score_blr[1])
        mse_arr[3].append(score_lars[1])

        rmse_arr[0].append(score_lr[2])
        rmse_arr[1].append(score_rr[2])
        rmse_arr[2].append(score_blr[2])
        rmse_arr[3].append(score_lars[2])

        r2_score_arr[0].append(score_lr[3])
        r2_score_arr[1].append(score_rr[3])
        r2_score_arr[2].append(score_blr[3])
        r2_score_arr[3].append(score_lars[3])

    model_names = ['LR', 'RR', 'BLR', 'LARS']
    print("------ Mean Evaluation Metrics -----")
    print("------------------------------------")
    print("Model\tMAE\tMSE\tRMSE\tR2Score")
    for i in range(4):
        print("{:s}\t{:7.5f} {:7.5f} {:7.5f} {:7.5f}".format(model_names[i],
                    np.mean(mae_arr[i]), np.mean(mse_arr[i]),
                    np.mean(rmse_arr[i]), np.mean(r2_score_arr[i])))