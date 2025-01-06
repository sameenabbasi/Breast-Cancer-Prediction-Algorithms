#importing everything needed
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

#got this info from website, loading the column names into list
columns = [
    'ID', 'Diagnosis', 'radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1', 'compactness1',
    'concavity1', 'concave_points1', 'symmetry1', 'fractal_dimension1', 'radius2', 'texture2', 'perimeter2',
    'area2', 'smoothness2', 'compactness2', 'concavity2', 'concave_points2', 'symmetry2', 'fractal_dimension2',
    'radius3', 'texture3', 'perimeter3', 'area3', 'smoothness3', 'compactness3', 'concavity3',
    'concave_points3', 'symmetry3', 'fractal_dimension3'
]

#loading the actual data
data = pd.read_csv('breast+cancer+wisconsin+diagnostic/wdbc.txt', header=None, names=columns)

# dropping id number since it is unneccesary and irrelevent
data = data.drop('ID', axis=1)

# encoding the target variable to binary (M = 1, B = 0)
data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})

# removing outliers using IQR
Q1 = data.quantile(0.05)  # need to figure out how much to remove
Q3 = data.quantile(0.95)  # need to figure out how much to remove
IQR = Q3 - Q1             # interquartile Range

# define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# filter out rows where any feature has an outlier
data_no_outliers = data[~((data < lower_bound) | (data > upper_bound)).any(axis=1)]

print(f"Original Data Shape: {data.shape}") # how many rows
print(f"Data Shape After Outlier Removal: {data_no_outliers.shape}") # how many rows once outliers are removed

# split the data into features and target
X = data.drop('Diagnosis', axis=1) #features
y = data['Diagnosis'] #target

# split into train/validation/test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# logistic regression - model 1
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# evaluate logistic regression 
y_pred_log = log_reg.predict(X_val)
y_pred_prob_log = log_reg.predict_proba(X_val)[:, 1]
y_pred_log_adjusted = (y_pred_prob_log >= 0.05).astype(int) #threshold i thought was best
print("Logistic Regression:")
print(classification_report(y_val, y_pred_log_adjusted))
print(f"ROC-AUC: {roc_auc_score(y_val, y_pred_prob_log):.2f}\n")

# svm - model 2
svm = SVC(probability=True)
svm.fit(X_train, y_train)

# evaluate svm
y_pred_svm = svm.predict(X_val)
y_pred_prob_svm = svm.predict_proba(X_val)[:, 1]
print("Support Vector Machine:")
print(classification_report(y_val, y_pred_svm))
print(f"ROC-AUC: {roc_auc_score(y_val, y_pred_prob_svm):.2f}\n")

# neural network 
nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
nn.fit(X_train, y_train)

# evaluate neural network
y_pred_nn = nn.predict(X_val)
y_pred_prob_nn = nn.predict_proba(X_val)[:, 1]
print("Neural Network:")
print(classification_report(y_val, y_pred_nn))
print(f"ROC-AUC: {roc_auc_score(y_val, y_pred_prob_nn):.2f}\n")

# final evaluation
# compare models on validation set
models = {
    'Logistic Regression': (y_pred_log_adjusted, y_pred_prob_log),
    'SVM': (y_pred_svm, y_pred_prob_svm),
    'Neural Network': (y_pred_nn, y_pred_prob_nn)
}

for model_name, (pred, prob) in models.items():
    print(f"{model_name}:\n")
    print(classification_report(y_val, pred))
    print(f"ROC-AUC: {roc_auc_score(y_val, prob):.2f}\n")
