import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skopt.space import Real, Categorical
import matplotlib.colors as mcolors
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
import torch
from torch.optim import Adam
from typing import Dict, Any
import sys


sarcasm_df = pd.read_csv("train-balanced-sarcasm.csv") #preprocessed  data
sarcasm_df



x_train, x_test, y_train, y_test = train_test_split(
    sarcasm_df["comment"], sarcasm_df["label"], test_size=0.3, random_state=42
)


# Feature Encoding and Model Training

# create tf-idf encoder
tf_idf = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2)
# encode training data
train_features = tf_idf.fit_transform(x_train)
# encode test data
test_features = tf_idf.transform(x_test)


# Hyperparameter_search
def hyperparameter_search(model, param_grid: Dict[str, Any], train_features, y_train):
    # Grid Search
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(train_features, y_train)
    best_parameters_grid_search = grid_search.best_params_
    
    # Random Search
    random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=15, cv=5)
    random_search.fit(train_features, y_train)
    best_parameters_random_search = random_search.best_params_
    
    # Bayesian Optimization
    bayes_search = BayesSearchCV(model, param_grid, n_iter=32, cv=5)
    bayes_search.fit(train_features, y_train)
    best_parameters_bayes_search = bayes_search.best_params_
    
    return best_parameters_grid_search, best_parameters_random_search, best_parameters_bayes_search


def train_on_best_params(model_class, best_params, train_features, y_train):
    """Train a model with the given best parameters."""
    model = model_class(**best_params)
    model.fit(train_features, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate a model and return accuracy, F1 score, precision, and recall."""
    predictions = model.predict(X_test)    
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    
    return accuracy, f1, precision, recall


# logistic regression model 


param_grid_log_reg = [
    {   'C': np.logspace(-4, 4, 20),
        'penalty': ['l1'],
        'solver': ['liblinear', 'saga']},
    {   'C': np.logspace(-4, 4, 20),
        'penalty': ['l2'],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga']}]


best_parameters_log_reg, best_params_random_log_reg, best_params_bayes_log_reg = hyperparameter_search(LogisticRegression(max_iter=10000), param_grid_log_reg, train_features, y_train)
log_reg_model_grid = LogisticRegression(max_iter=10000, **best_parameters_log_reg).fit(train_features, y_train)
accuracy_log_reg, f1_log_reg, precision_log_reg, recall_log_reg = evaluate_model(log_reg_model_grid, test_features, y_test)
print("Results for Logistic Regression Model:")
print(f"Accuracy: {accuracy_log_reg:.2f}, F1: {f1_log_reg:.2f}, Precision: {precision_log_reg:.2f}, Recall: {recall_log_reg:.2f}\n")




# Naive Bayes 
param_grid_nb = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'fit_prior': [True, False]
}

best_parameters_nb, best_params_random_nb, best_params_bayes_nb = hyperparameter_search(MultinomialNB(), param_grid_nb, train_features, y_train)
nb_model_grid = MultinomialNB(**best_parameters_nb).fit(train_features, y_train)
accuracy_nb, f1_nb, precision_nb, recall_nb = evaluate_model(nb_model_grid, test_features, y_test)

print("Results for MultinomialNB Model:")
print(f"Accuracy: {accuracy_nb:.2f}, F1: {f1_nb:.2f}, Precision: {precision_nb:.2f}, Recall: {recall_nb:.2f}\n")



# K-Nearest Neighbors
param_grid_knn = {
    'n_neighbors': list(range(1, 30)),
    'p': [1, 2]
}
best_parameters_knn, best_params_random_knn, best_params_bayes_knn = hyperparameter_search(KNeighborsClassifier(), param_grid_knn, train_features, y_train)
knn_model_grid = KNeighborsClassifier(**best_parameters_knn).fit(train_features, y_train)
accuracy_knn, f1_knn, precision_knn, recall_knn = evaluate_model(knn_model_grid, test_features, y_test)
print("Results for K-Nearest Neighbors Model:")
print(f"Accuracy: {accuracy_knn:.2f}, F1: {f1_knn:.2f}, Precision: {precision_knn:.2f}, Recall: {recall_knn:.2f}\n")




# SVM
param_grid_svm = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
}
best_parameters_svm, best_params_random_svm, best_params_bayes_svm = hyperparameter_search(SVC(), param_grid_svm, train_features, y_train)
svm_model_grid = SVC(**best_parameters_svm).fit(train_features, y_train)
accuracy_svm, f1_svm, precision_svm, recall_svm = evaluate_model(svm_model_grid, test_features, y_test)
print("Results for Support Vector Machine Model:")
print(f"Accuracy: {accuracy_svm:.2f}, F1: {f1_svm:.2f}, Precision: {precision_svm:.2f}, Recall: {recall_svm:.2f}\n")




# Decision Tree
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 5, 10, 15, 20, 25],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
best_parameters_dt, best_params_random_dt, best_params_bayes_dt = hyperparameter_search(DecisionTreeClassifier(), param_grid_dt, train_features, y_train)
dt_model_grid = DecisionTreeClassifier(**best_parameters_dt).fit(train_features, y_train)
accuracy_dt, f1_dt, precision_dt, recall_dt = evaluate_model(dt_model_grid, test_features, y_test)
print("Results for Decision Tree Model:")
print(f"Accuracy: {accuracy_dt:.2f}, F1: {f1_dt:.2f}, Precision: {precision_dt:.2f}, Recall: {recall_dt:.2f}\n")

# Visualizing the decision tree
np.set_printoptions(threshold=10)
sys.setrecursionlimit(10000)  
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(train_features, y_train)
plt.figure(figsize=(40,20))
plot_tree(dt, filled=True, feature_names=tf_idf.get_feature_names(), class_names=["Not Sarcastic", "Sarcastic"], rounded=True, fontsize=10)
plt.show()



# Bagging Decision Tree 
param_grid_bagging_dt = {
    'n_estimators': [10, 20, 30, 40, 50],
    'max_samples': [0.5, 1.0],
    'max_features': [0.5, 1.0],
    'bootstrap': [True, False],
    'bootstrap_features': [True, False]
}
best_parameters_bagging_dt, best_params_random_bagging_dt, best_params_bayes_bagging_dt = hyperparameter_search(BaggingClassifier(), param_grid_bagging_dt, train_features, y_train)
bagging_dt_model_grid = BaggingClassifier(**best_parameters_bagging_dt).fit(train_features, y_train)
accuracy_bagging_dt, f1_bagging_dt, precision_bagging_dt, recall_bagging_dt = evaluate_model(bagging_dt_model_grid, test_features, y_test)
print("Results for Bagging Decision Tree Model:")
print(f"Accuracy: {accuracy_bagging_dt:.2f}, F1: {f1_bagging_dt:.2f}, Precision: {precision_bagging_dt:.2f}, Recall: {recall_bagging_dt:.2f}\n")



# Boosted Decision Tree
param_grid_adaboost = {
    'n_estimators': [10, 50, 100, 150, 200],
    'learning_rate': [0.001, 0.01, 0.1, 1, 10],
    'algorithm': ['SAMME', 'SAMME.R']
}
best_parameters_adaboost, best_params_random_adaboost, best_params_bayes_adaboost = hyperparameter_search(AdaBoostClassifier(), param_grid_adaboost, train_features, y_train)
adaboost_model_grid = AdaBoostClassifier(**best_parameters_adaboost).fit(train_features, y_train)
accuracy_adaboost, f1_adaboost, precision_adaboost, recall_adaboost = evaluate_model(adaboost_model_grid, test_features, y_test)
print("Results for AdaBoostClassifier Model:")
print(f"Accuracy: {accuracy_adaboost:.2f}, F1: {f1_adaboost:.2f}, Precision: {precision_adaboost:.2f}, Recall: {recall_adaboost:.2f}\n")




# Random Forest
param_grid_rf = {
    'n_estimators': [10, 50, 100, 150, 200],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
best_parameters_rf, best_params_random_rf, best_params_bayes_rf = hyperparameter_search(RandomForestClassifier(), param_grid_rf, train_features, y_train)
rf_model_grid = train_on_best_params(RandomForestClassifier, best_parameters_rf, train_features, y_train)
rf_model_random = train_on_best_params(RandomForestClassifier, best_params_random_rf, train_features, y_train)
rf_model_bayes = train_on_best_params(RandomForestClassifier, dict(best_params_bayes_rf), train_features, y_train)  

print("Best Params from Grid Search:", best_parameters_rf)
print("Best Params from Random Search:", best_params_random_rf)
print("Best Params from Bayesian Optimization:", best_params_bayes_rf)

accuracy_grid, f1_grid, precision_grid, recall_grid = evaluate_model(rf_model_grid, test_features, y_test)
accuracy_random, f1_random, precision_random, recall_random = evaluate_model(rf_model_random, test_features, y_test)
accuracy_bayes, f1_bayes, precision_bayes, recall_bayes = evaluate_model(rf_model_bayes, test_features, y_test)

print("Results for Grid Search Model:")
print(f"Accuracy: {accuracy_grid:.2f}, F1: {f1_grid:.2f}, Precision: {precision_grid:.2f}, Recall: {recall_grid:.2f}\n")

print("Results for Random Search Model:")
print(f"Accuracy: {accuracy_random:.2f}, F1: {f1_random:.2f}, Precision: {precision_random:.2f}, Recall: {recall_random:.2f}\n")

print("Results for Bayesian Optimization Model:")
print(f"Accuracy: {accuracy_bayes:.2f}, F1: {f1_bayes:.2f}, Precision: {precision_bayes:.2f}, Recall: {recall_bayes:.2f}\n")




# Voting Classifier
classifiers = [
    ('log_reg', LogisticRegression(max_iter=10000, **best_parameters_log_reg)),
    ('nb', MultinomialNB(**best_parameters_nb)),
    ('dt', DecisionTreeClassifier(**best_parameters_dt)),
    ('bagging_dt', BaggingClassifier(**best_parameters_bagging_dt)),
]
param_grid_voting = {
    'voting': ['hard', 'soft'],
}
best_parameters_voting, best_params_random_voting, best_params_bayes_voting = hyperparameter_search(VotingClassifier(estimators=classifiers), param_grid_voting, train_features, y_train)
voting_model_grid = VotingClassifier(estimators=classifiers, **best_parameters_voting).fit(train_features, y_train)
accuracy_voting, f1_voting, precision_voting, recall_voting = evaluate_model(voting_model_grid, test_features, y_test)
print("Results for Voting Classifier Model:")
print(f"Accuracy: {accuracy_voting:.2f}, F1: {f1_voting:.2f}, Precision: {precision_voting:.2f}, Recall: {recall_voting:.2f}\n")



# Predicted labels for each model
pred_labels_log_reg = log_reg_model_grid.predict(test_features)
pred_labels_naive_bayes = nb_model_grid.predict(test_features)
pred_labels_knn=knn_model_grid.predict(test_features)
pred_labels_svc=svm_model_grid .predict(test_features)
pred_labels_dt = dt_model_grid.predict(test_features)
pred_labels_bagging_dt = bagging_dt_model_grid.predict(test_features)
pred_labels_boosted_dt = adaboost_model_grid.predict(test_features)
pred_labels_rf = rf_model_grid.predict(test_features)
pred_labels_voting_clf = voting_model_grid.predict(test_features)
# Add these predictions to your list of accuracy scores
scores = [accuracy_score(y_test, pred_labels_log_reg), 
          accuracy_score(y_test, pred_labels_naive_bayes),
          accuracy_score(y_test, pred_labels_knn),
          accuracy_score(y_test, pred_labels_svc),
          accuracy_score(y_test, pred_labels_dt),
          accuracy_score(y_test, pred_labels_bagging_dt),
          accuracy_score(y_test, pred_labels_boosted_dt),
          accuracy_score(y_test, pred_labels_rf),
          accuracy_score(y_test, pred_labels_voting_clf)]

classifiers = ['Logistic Regression', 'Naive Bayes', 'K-Nearest Neighbours', 'Support Vector Machine', 'Decision Tree', 'Bagging Decision Tree', 'Boosted Decision Tree', 'Random Forest', 'Voting Classifier']

cmap = mcolors.LinearSegmentedColormap.from_list("n",['#add8e6','#00008b'])

plt.figure(figsize=(10, 5))
bars = plt.barh(classifiers, scores, color=cmap(scores))

plt.xlabel('Accuracy Score')
plt.title('Comparison of Different Classifiers')
plt.xlim(0, 1.0)

plt.savefig('classifier_comparison.png', bbox_inches='tight')

plt.show()


