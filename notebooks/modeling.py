import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import RandomOverSampler
import pickle
from sklearn import metrics
from sklearn.metrics import log_loss, precision_recall_fscore_support, accuracy_score, f1_score, confusion_matrix, mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load

df = pd.read_csv(r"C:\Users\LocalAdmin\Desktop\DAB_vse\DAB_2_s\DataX\project\Pinguins\data\penguins.csv")
df.info()

#initial preprocess based on data exploration
for i in ["year","species","island","sex"]:
      df[i] = df[i].astype('category')

df = df.drop(['year'], axis = 1)


#X_train.info()
#split
#we will use standard 0.7,0.15,0.15 split
#define splitter fun
def get_data_splits(X, y, train_size=0.7):
    """Generate balanced data splits."""
    X_train, X_, y_train, y_ = train_test_split(
        X, y, train_size=train_size, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_, y_, train_size=0.5, stratify=y_)
    return X_train, X_val, X_test, y_train, y_val, y_test

def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)

le = LabelEncoder()

set_seeds()
X_train, X_val, X_test, y_train, y_val, y_test = \
    get_data_splits(X=df.loc[:, df.columns != 'species'], y=le.fit_transform(df.species))

#oversample
# Class weights
counts = np.bincount(y_train)
class_weights = {i: 1.0/count for i, count in enumerate(counts)}
print (f"class counts: {counts},\nclass weights: {class_weights}")

# Oversample (training set)
oversample = RandomOverSampler(sampling_strategy="all")
X_over, y_over = oversample.fit_resample(X_train, y_train)

# Class weights
counts = np.bincount(y_over)
class_weights = {i: 1.0/count for i, count in enumerate(counts)}
print (f"class counts: {counts},\nclass weights: {class_weights}")


#Data save to pickle format in intermediate format to be further transformed in train and test
with open(r"inter_data.pickle", "wb") as output_file:
    pickle.dump([X_over, y_over, X_test, y_test, X_val, y_val], output_file)


## Definition of fce to process train data and create dict with mean/modus to use
def process_train(df):
    # Make a copy of the input DataFrame
    df_copy = df.copy()

    # Impute categorical columns with mode
    cat_cols = df_copy.select_dtypes(include='category').columns
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df_copy[cat_cols] = cat_imputer.fit_transform(df_copy[cat_cols])
    cat_imputer_dict = {col: cat_imputer.statistics_[i] for i, col in enumerate(cat_cols)}

    # Impute numerical columns with mean
    num_cols = df_copy.select_dtypes(include='number').columns
    num_imputer = SimpleImputer(strategy='mean')
    df_copy[num_cols] = num_imputer.fit_transform(df_copy[num_cols])
    num_imputer_dict = {col: num_imputer.statistics_[i] for i, col in enumerate(num_cols)}

    # Encode sex column to binary value
    le = LabelEncoder()
    df_copy['sex'] = le.fit_transform(df_copy['sex'])

    # One-hot encode island column
    ohe = OneHotEncoder(drop='first')
    island_encoded = ohe.fit_transform(df_copy[['island']])
    island_encoded_df = pd.DataFrame(island_encoded.toarray())
    island_encoded_df.columns = ['island_{}'.format(val) for val in ohe.categories_[0][1:]]
    df_copy = pd.concat([df_copy.drop('island', axis=1), island_encoded_df], axis=1)

    return df_copy, cat_imputer_dict, num_imputer_dict


X_over_p, cat_imputer_dict, num_imputer_dict = process_train(X_over)

# Definition of fce to process val data with mean/modus from train
def process_val(new_df, cat_imputer_dict, num_imputer_dict):
    # Create a copy of the dataframe to avoid modifying the original
    new_df_copy = new_df.copy()

    # Impute categorical columns with saved mode
    new_cat_cols = new_df_copy.select_dtypes(include='category').columns
    for col in new_cat_cols:
        if col in cat_imputer_dict:
            new_df_copy[col].fillna(cat_imputer_dict[col], inplace=True)

    # Impute numerical columns with saved mean
    new_num_cols = new_df_copy.select_dtypes(include='number').columns
    for col in new_num_cols:
        if col in num_imputer_dict:
            new_df_copy[col].fillna(num_imputer_dict[col], inplace=True)

    # Encode sex column to binary value
    le = LabelEncoder()
    new_df_copy['sex'] = le.fit_transform(new_df_copy['sex'])

    # One-hot encode island column
    ohe = OneHotEncoder(drop='first')
    island_encoded = ohe.fit_transform(new_df_copy[['island']])
    island_encoded_df = pd.DataFrame(island_encoded.toarray())
    island_encoded_df.columns = ['island_{}'.format(val) for val in ohe.categories_[0][1:]]
    new_df_copy.reset_index(drop=True, inplace=True) # reset index before concatenation
    new_df_processed = pd.concat([new_df_copy.drop('island', axis=1), island_encoded_df], axis=1)

    return new_df_processed

x_val_p = process_val(X_val, cat_imputer_dict, num_imputer_dict)
x_test_p = process_val(X_test, cat_imputer_dict, num_imputer_dict)

#Data save to pickle format in processed final format used for training and val
with open(r"processed_data.pickle", "wb") as output_file:
    pickle.dump([X_over_p, y_over, x_test_p, y_test, x_val_p, y_val], output_file)

##Decition tree
# using grid search, we set up hyperparametres, same in further models
hyper_grid = {
    'max_depth': [2, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

tree_model_cv = DecisionTreeClassifier(random_state=42)

grid_search = GridSearchCV(estimator = tree_model_cv, 
                          param_grid = hyper_grid, 
                          cv = 5, 
                          n_jobs = -1, 
                          verbose = 2
                          )

grid_search.fit(X_over_p, y_over)
print(grid_search.best_params_)

model_cv_opt= grid_search.best_estimator_

y_pred_cv = model_cv_opt.predict(x_val_p)
y_train_cv= model_cv_opt.predict(X_over_p)

rmse_model_cv = mean_squared_error(y_val, y_pred_cv, squared = False)
print(rmse_model_cv)

accuracy = metrics.accuracy_score(y_over, y_train_cv)
print(f"Accuracy score: {accuracy:.3f}")

# Calculate the accuracy score on the validation set
accuracy = metrics.accuracy_score(y_val, y_pred_cv)
print(f"Accuracy score: {accuracy:.3f}")

#test predictions
y_test_pred = model_cv_opt.predict(x_test_p)

accuracy = metrics.accuracy_score(y_test, y_test_pred)
print(f"Accuracy score: {accuracy:.3f}")

precision = metrics.precision_score(y_test, y_test_pred, average='macro')
print(f"Precision score: {precision:.3f}")

recall = metrics.recall_score(y_test, y_test_pred, average='macro')
print(f"Recall score: {accuracy:.3f}")

F1 = 2 * (precision * recall) / (precision + recall)
print(f"F1 score: {F1:.3f}")

dump(model_cv_opt, 'Decision_tree.h5')

##KNN
#defined pipeline steps with KNeighborsClassifier
steps = [
    ('scaler', MinMaxScaler()),
    ('knn', KNeighborsClassifier())
]

# ini pipeline
pipeline = Pipeline(steps)

#actual training
pipeline.fit(X_over_p, y_over)

y_val_pred = pipeline.predict(x_val_p)

accuracy = metrics.accuracy_score(y_val, y_val_pred)
print(f"Accuracy score: {accuracy:.3f}")

y_test_pred = pipeline.predict(x_test_p)

accuracy = metrics.accuracy_score(y_test, y_test_pred)
print(f"Accuracy score: {accuracy:.3f}")

precision = metrics.precision_score(y_test, y_test_pred, average='macro')
print(f"Precision score: {precision:.3f}")

recall = metrics.recall_score(y_test, y_test_pred, average='macro')
print(f"Recall score: {accuracy:.3f}")

F1 = 2 * (precision * recall) / (precision + recall)
print(f"F1 score: {F1:.3f}")


dump(pipeline, 'KNN.h5')

##Random Forrest
hyper_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_cv = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator = rf_cv, param_grid = hyper_grid,
                           cv = 5, n_jobs = -1, verbose = 2)

grid_search.fit(X_over_p, y_over)

print(grid_search.best_params_)

forest_model_opt= grid_search.best_estimator_ #best_estimator_

y_pred_cv = forest_model_opt.predict(x_val_p)

accuracy = metrics.accuracy_score(y_val, y_pred_cv)
print(f"Accuracy score: {accuracy:.3f}")

y_test_cv = forest_model_opt.predict(x_test_p)

accuracy = metrics.accuracy_score(y_test, y_test_cv)
print(f"Accuracy score: {accuracy:.3f}")

precision = metrics.precision_score(y_test, y_test_pred, average='macro')
print(f"Precision score: {precision:.3f}")

recall = metrics.recall_score(y_test, y_test_pred, average='macro')
print(f"Recall score: {accuracy:.3f}")

F1 = 2 * (precision * recall) / (precision + recall)
print(f"F1 score: {F1:.3f}")

#feature_importances
importances_rf = pd.Series(forest_model_opt.feature_importances_, index = X_over_p.columns)
# Sort importances_rf
sorted_importances_rf = importances_rf.sort_values() 
# Make a horizontal bar plot
sorted_importances_rf.plot(kind='barh', color='lightgreen') 
plt.show()

dump(forest_model_opt, 'RF.h5')

##XGB

# create your XGBoost model
xgb = XGBClassifier(seed=42)

# define the hyperparameters
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [2, 3, 5, 7],
    'learning_rate': [0.2, 0.1, 0.01, 0.001],
    'tree_method': ['auto', 'exact', 'approx'],
}

# finding the best hyperparameters
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, verbose = 1)
grid_search.fit(X_over_p, y_over)

# print the best hyperparameters found by grid search
print(grid_search.best_params_)

xgb_model = grid_search.best_estimator_
y_val_cv = xgb_model.predict(x_val_p)

accuracy = metrics.accuracy_score(y_val, y_val_cv)
print(f"Accuracy score: {accuracy:.5f}")

y_test_cv = xgb_model.predict(x_test_p)

accuracy = metrics.accuracy_score(y_test, y_test_cv)
print(f"Accuracy score: {accuracy:.3f}")

precision = metrics.precision_score(y_test, y_test_pred, average='macro')
print(f"Precision score: {precision:.3f}")

recall = metrics.recall_score(y_test, y_test_pred, average='macro')
print(f"Recall score: {accuracy:.3f}")

F1 = 2 * (precision * recall) / (precision + recall)
print(f"F1 score: {F1:.3f}")

### 1 ACC test

#feature_importances
importances_rf = pd.Series(xgb_model.feature_importances_, index = X_over_p.columns)
# Sort importances_rf
sorted_importances_rf = importances_rf.sort_values() 
# Make a horizontal bar plot
sorted_importances_rf.plot(kind='barh', color='lightgreen') 
plt.show()

dump(xgb_model, 'XGB.h5')