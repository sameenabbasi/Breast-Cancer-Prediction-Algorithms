import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score, recall_score, precision_score, f1_score
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

#loading the dataset
data = pd.read_csv('breast+cancer+wisconsin+diagnostic/wdbc.txt', header=None, names=columns)

# dropping id number since it is unneccesary 
data = data.drop('ID', axis=1)

# encoding the target variable to binary (M = 1, B = 0)
data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})



#box plot code
features_to_plot = ['radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1', 'compactness1',
    'concavity1', 'symmetry1', 'fractal_dimension1']

# removing outliers using IQR
Q1 = data.quantile(0.05)  # changed quartile to 0.05 to include more
Q3 = data.quantile(0.95)  # changed quartile to 0.95 to include more outliers
IQR = Q3 - Q1             # interquartile range

# define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

#box plots before removing outliers
plt.figure(figsize=(10, 6))
for i, feature in enumerate(features_to_plot, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(data=data, x=feature, color='skyblue')
    plt.title(f"Boxplot of {feature}")
    plt.xlabel("")
plt.tight_layout()
plt.show()

# filter out rows where any feature has an outlier
data_no_outliers = data[~((data < lower_bound) | (data > upper_bound)).any(axis=1)]

print(f"Original Data Shape: {data.shape}") # how many rows
print(f"Data Shape After Outlier Removal: {data_no_outliers.shape}") # how many rows once outliers are removed

# split the data into features and target
X = data.drop('Diagnosis', axis=1) #features
y = data['Diagnosis'] #target

# split it into train/vlidation/test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)





# correlation matrix
print(X_train.shape)  # Should output (398, 30)
feature_columns = columns[2:]  # Skip 'ID' and 'Diagnosis'
corr_matrix = pd.DataFrame(X_train, columns = feature_columns).corr()
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title('Feature Correlation Matrix')
plt.show()




# select a subset of features for better visualization 
subset_features = ['radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1', 'compactness1',
    'concavity1',  'symmetry1', 'fractal_dimension1'] #selected the first set
subset_data = pd.DataFrame(X_train, columns=feature_columns)[subset_features]
subset_data['Diagnosis'] = y_train.reset_index(drop=True)  # add target variable for coloring

# create the scatter plot matrix
sns.pairplot(subset_data, hue='Diagnosis', diag_kind='kde', corner=True)
plt.suptitle("Scatter Plot Matrix for Selected Features", y=1.02)
plt.show()





# logistic regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# evaluate logistic regression
y_pred_log = log_reg.predict(X_val)
y_pred_prob_log = log_reg.predict_proba(X_val)[:, 1]

# test thresholds and calculate recall
thresholds = np.arange(0.0, 1.1, 0.05)
recalls = []

for threshold in thresholds:
    # convert probabilities to binary predictions based on the threshold
    y_pred_threshold = (y_pred_prob_log >= threshold).astype(int)
    
    # calculate recall for this threshold
    recall = recall_score(y_val, y_pred_threshold)
    recalls.append(recall)


# plot recall scores for different thresholds
import matplotlib.pyplot as plt

plt.plot(thresholds, recalls, marker='o')
plt.title('Recall vs Threshold')
plt.xlabel('Threshold')
plt.ylabel('Recall')
plt.grid(True)
plt.show()




# predict probabilities for the validation set
y_pred_prob_log = log_reg.predict_proba(X_val)[:, 1]

# test various thresholds and calculate recall, precision, and f1-score
thresholds = np.arange(0.0, 1.1, 0.05)
recalls = []
precisions = []
f1_scores = []

for threshold in thresholds:
    # convert probabilities to binary predictions based on the threshold
    y_pred_threshold = (y_pred_prob_log >= threshold).astype(int)
    
    # calculate recall, precision, and f1-score for this threshold
    recall = recall_score(y_val, y_pred_threshold)
    precision = precision_score(y_val, y_pred_threshold)
    f1 = f1_score(y_val, y_pred_threshold)
    
    recalls.append(recall)
    precisions.append(precision)
    f1_scores.append(f1)

# plot recall, precision, and f1 score for different thresholds
plt.figure(figsize=(10, 6))

plt.plot(thresholds, recalls, marker='o', label='Recall', color='blue')
plt.plot(thresholds, precisions, marker='x', label='Precision', color='green')
plt.plot(thresholds, f1_scores, marker='s', label='F1-Score', color='red')

plt.title('Recall, Precision, and F1-Score vs Threshold')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.show()