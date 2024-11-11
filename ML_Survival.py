from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.preprocessing import OneHotEncoder, encode_categorical
from sksurv.ensemble import RandomSurvivalForest
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


random_state = 42


features = pd.read_csv('Median_Vals/Features_Survival_Test.txt', sep = ' ', index_col = 0)
yval = pd.read_csv('Median_Vals/TEST_Fluxes_Survival.txt', index_col = 0)

y = yval['flux_line']

X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.25, random_state=42)

#kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
n_features = features.shape[1]
kernel = RBF(length_scale=[1.0] * n_features, length_scale_bounds=(1e-3, 1e3))

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)


# estimator = RandomSurvivalForest(n_estimators=1000, 
#                                  min_samples_split=10, 
#                                  min_samples_leaf=15, 
#                                  n_jobs=-1, 
#                                  random_state=random_state)
#estimator.fit(X_train, y_train)