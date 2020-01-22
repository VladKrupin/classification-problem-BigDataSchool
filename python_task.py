import pandas as pd
import numpy as np

dataset=pd.read_csv('DR_Demo_Lending_Club_reduced.csv')

dataset.info()
info=dataset.describe()
dataset.columns
dataset=dataset.drop(['Id','mths_since_last_delinq','mths_since_last_record', 
                      'initial_list_status','zip_code', 'addr_state', 'pymnt_plan'], axis=1)

mis_val=['annual_inc','open_acc','revol_util','total_acc','open_acc','revol_util','total_acc']
for i in mis_val:
    dataset.loc[(dataset['is_bad']==0) & (dataset[i].isnull()), i]=dataset[dataset['is_bad']==0][i].mean()
    dataset.loc[(dataset['is_bad']==1) & (dataset[i].isnull()), i]=dataset[dataset['is_bad']==1][i].mean()

mis_val2=['delinq_2yrs','inq_last_6mths','pub_rec','collections_12_mths_ex_med']
for i in mis_val2:
    dataset.loc[(dataset['is_bad']==0) & (dataset[i].isnull()), i]=dataset[dataset['is_bad']==0][i].value_counts()[0]
    dataset.loc[(dataset['is_bad']==1) & (dataset[i].isnull()), i]=dataset[dataset['is_bad']==1][i].value_counts()[0]

cat_val=['home_ownership', 'verification_status', 'purpose_cat', 'policy_code']
dataset['emp_length']=np.where(dataset['emp_length']=='na', 0,dataset['emp_length'])

for i in cat_val:
    dataset=pd.concat((dataset, pd.get_dummies(dataset[i], drop_first=True)), axis=1)
    dataset=dataset.drop([i], axis=1)

Y=dataset['is_bad']
X=dataset
X=X.drop(['is_bad'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, random_state=0, test_size=0.2)

X = pd.concat([X_train, Y_train], axis=1)

not_bad=X[X.is_bad==0]
bad=X[X.is_bad==1]

from sklearn.utils import resample
bad_unsampled = resample(bad,
                          replace=True,
                          n_samples=len(not_bad),
                          random_state=27)

upsampled = pd.concat([not_bad, bad_unsampled])

X_train=upsampled.iloc[:,:-1]
Y_train=upsampled.iloc[:,-1]

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

from xgboost import XGBClassifier
classifier=XGBClassifier(random_state=0)

parameters = {
    'max_depth': range (2, 10, 1),
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05]
}

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(
    estimator=classifier,
    param_grid=parameters,
    scoring = 'roc_auc',
    n_jobs = -1,
    cv = 10,
    verbose=True
)

grid_search.fit(X_train, Y_train)
grid_search.best_estimator_
grid_search.best_score_

classifier=XGBClassifier(random_state=0,n_estimators=180,learning_rate=0.1,max_depth=2)
classifier.fit(X_train, Y_train)
y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_auc_score, accuracy_score
cm=confusion_matrix(Y_test, y_pred)

print('accuracy=',accuracy_score(Y_test, y_pred))
print('recall=',recall_score(Y_test, y_pred))
print('precision=',precision_score(Y_test, y_pred))
print('roc_auc=',roc_auc_score(Y_test, y_pred))
