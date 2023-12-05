import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score, plot_confusion_matrix, plot_precision_recall_curve

# Load data and drop unnecessary column 'ID'
credit_risk = pd.read_csv("UCI_credit_card.csv")
df = credit_risk.drop(columns=["ID"])

# Prepare independent and dependent features
X = StandardScaler().fit_transform(df.drop(['default.payment.next.month'], axis=1))
y = df['default.payment.next.month']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Apply SMOTE for oversampling
print("Before oversampling:", Counter(y_train))
X_train, y_train = SMOTE().fit_resample(X_train, y_train)
print("After oversampling:", Counter(y_train))

# Logistic Regression
logit = LogisticRegression()
logit.fit(X_train, y_train)
pred_logit = logit.predict(X_test)

# Evaluate Logistic Regression model
print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, pred_logit))
print(classification_report(y_test, pred_logit))
plot_confusion_matrix(logit, X_test, y_test, cmap="Blues_r")
plt.show()
plot_precision_recall_curve(logit, X_test, y_test)
plt.show()

# Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

# Evaluate Random Forest Classifier model
print("\nRandom Forest Classifier:")
print("Accuracy:", accuracy_score(y_test, pred_rf))
print(classification_report(y_test, pred_rf))

# XGB Classifier
xgb_clf = xgb.XGBClassifier()
xgb_clf.fit(X_train, y_train)
xgb_predict = xgb_clf.predict(X_test)

# Evaluate XGB Classifier model
print("\nXGB Classifier:")
print("Accuracy:", accuracy_score(y_test, xgb_predict))
print(classification_report(y_test, xgb_predict))
plot_confusion_matrix(xgb_clf, X_test, y_test, cmap="Blues_r")
plt.show()
plot_precision_recall_curve(xgb_clf, X_test, y_test)
plt.show()

# Hyperparameter optimization using RandomizedSearchCV for XGB Classifier
params = {
    "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
    "reg_lambda": [3, 4, 5, 6, 8, 10, 12, 15],
    "subsample": [0.3, 0.4, 0.5, 0.7, 0.9, 1.1, 1.3],
    "colsample_bytree": [0.3, 0.4, 0.5, 0.7]
}

random_search = RandomizedSearchCV(xgb_clf, param_distributions=params, n_iter=5, scoring='roc_auc', n_jobs=-1, cv=5, verbose=3)
print("Fitting RandomizedSearchCV")
random_search.fit(X_train, y_train)

# Find the best estimators and parameters
print("Best Estimator:", random_search.best_estimator_)
print("Best Parameters:", random_search.best_params_)

# Create XGB Classifier with best parameters
classifierXGB = xgb.XGBClassifier(
    objective='binary:logistic',
    gamma=random_search.best_params_['gamma'],
    learning_rate=random_search.best_params_['learning_rate'],
    max_depth=random_search.best_params_['max_depth'],
    reg_lambda=random_search.best_params_['reg_lambda'],
    min_child_weight=random_search.best_params_['min_child_weight'],
    subsample=random_search.best_params_['subsample'],
    colsample_bytree=random_search.best_params_['colsample_bytree'],
    use_label_encoder=False
)

# Fit the model and predict
classifierXGB.fit(X_train, y_train)
y_pred = classifierXGB.predict(X_test)

# Cross-validation scores
cv_scores = cross_val_score(classifierXGB, X, y, cv=10)
print("\nCross-Validation Scores:", cv_scores)
print("Mean of the scores:", cv_scores.mean())
