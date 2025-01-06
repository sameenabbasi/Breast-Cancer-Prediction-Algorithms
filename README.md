This code explores the effectiveness of three machine learning models—Logistic Regression, Support Vector Machine (SVM), and Neural Networks—in detecting breast cancer using the Breast Cancer Wisconsin (Diagnostic) dataset. The goal is to evaluate the performance of these models in accurately classifying malignant and benign tumours. The dataset contains various continuous features such as radius, texture,
perimeter, and smoothness measurements, and is used to train and test the models. By comparing the models based on performance metrics like precision, recall, F1-
score, and ROC-AUC, I aim to identify the most reliable method for breast cancer detection.

The results indicate strong performance for all three models across various metrics,
demonstrating their suitability for the classification task. Logistic Regression achieved a
perfect ROC-AUC score of 1.00, reflecting its ability to perfectly separate classes based
on predicted probabilities.
The SVM slightly outperformed the others in terms of overall accuracy (96%) and
macro-average precision (0.97), suggesting its robustness in identifying patterns within
the data. The Neural Network also performed well, achieving precision and recall values of 0.96 and
0.95 for classes 0 and 1, respectively. Its ROC-AUC score of 0.99 highlights its nearperfect
ability to distinguish between classes.

Among the models, I prefer Logistic Regression due to its perfect ROC-AUC score and
superior recall for class 1, the malignant cases. In this context, it is much more
important to identify all cases of cancer, even at the cost of some false positives, to
ensure maximum safety. Missing a diagnosis (false negatives) could be detrimental,
making high recall critical. While the SVM and Neural Network offer strong
performance, Logistic Regression aligns best with the task’s goal of maximizing true
positives, even if it means allowing some false positives as a precautionary measure.
