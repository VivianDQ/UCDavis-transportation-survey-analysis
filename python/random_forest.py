#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import time
import collections

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier as boosting
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)


# In[3]:


# Load the data
# data_dic = pd.read_csv("data/cts1819_data_dictionary.csv")
data = pd.read_csv("data/CTS_noID_1819.csv")


# In[9]:


np.random.seed(1)


# In[10]:


class Classify:
    def __init__(self):
        # self.data = data
        self.X = None
        self.y = None
        
    def process_data(self, col, data):
        xname = ['primary_role',
                'op_travel_wasted',
                'op_like_biking',
                'op_eco_concern',
                'op_like_driving',
                'op_need_car',
                'op_schedule_transit',
                'op_feel_safe',
                'op_like_transit',
                'op_need_own_car',
                'op_limit_driving',
                'op_smartphone',
                'op_dress_professional',
                'op_travel_stress']
        data[col].replace(['Personal bike', 'Bike share (e.g. JUMP)'], 'bike', inplace = True)
        data[col].replace(['Bus and/or shuttle', 'Train and/or light rail'], 'bus', inplace = True)
        data[col].replace(['Lyft, Uber, or other ride-hailing service', 'Carpool and/or vanpool with others', 'Drive alone in a car (or other vehicle)', 'Get dropped off by a friend of family'], 'drive', inplace = True)
        data[col].replace(['Walk (or wheelchair)', 'Skate, skateboard, or scooter', 'Other:'], 'other', inplace = True)
        
        if col == 'lastmile_bus':
            data[col].replace(['drive'], 'other', inplace = True)
        if col == 'lastmile_train':
            data[col].replace(['Get dropped off by a friend or family'], 'drive', inplace = True)
            # data[col].replace(['drive'], 'drive&bus', inplace = True)
            # data[col].replace(['bus'], 'drive&bus', inplace = True)
        if col == 'firstmile_train':
            data[col].replace(['bus'], 'other', inplace = True)
        
        data.replace(['Strongly agree'], 5, inplace = True)
        data.replace(['Somewhat agree'], 4, inplace = True)
        data.replace(['Neither agree nor disagree'], 3, inplace = True)
        data.replace(['Somewhat disagree'], 2, inplace = True)
        data.replace(['Strongly disagree'], 1, inplace = True)
        
        data['primary_role'].replace(['Undergraduate student (including Post-baccalaureate)'], 'undergra', inplace = True)
        data['primary_role'].replace(['Graduate student'], 'gra', inplace = True)
        data['primary_role'].replace(['Faculty'], 'fac', inplace = True)
        data['primary_role'].replace(["I'm no longer affiliated with UC Davis", 'Other:'], 'other', inplace = True)
        data['primary_role'].replace(['Visiting scholar', 'Staff', 'Post doc'], 'staff', inplace = True)
        self.X, self.y = data.dropna(subset=[col] + xname)[xname], data.dropna(subset=[col] + xname)[col]
        self.X = pd.get_dummies(self.X)
        return self.X, self.y
    
    def summary(self):
        print(collections.Counter(self.y))

    def training_process(self, train_X, train_y, parameters, model, criteria = 'balanced_accuracy'):
        '''
        This function trains several models based on the 'neg_mean_squared_error' (negative mse) from cross-validation on training set 
        and return trained model, best hyperparameters and performance in training 
        so that it can be compared with test performance.
        The default setting is for classification problems.

        Parameters:
        ==========================================
        train_X: training features, will be fed in sklearn models
        train_y: training labels, will be fed in sklearn models
        parameters: a dictionary of hyperparameters to choose from. e.g. {[parameter name]:[list of choices]}
        model: the basic sklearn model. In this homework, svc()
        '''

        t1 = time.time()
        clf = GridSearchCV(model, parameters, cv=5, scoring=criteria, n_jobs=-1)
        clf.fit(train_X, train_y)
        t2 = time.time()

        print('Training finished! Time usage {}'.format(t2-t1))
        best_parameters = clf.cv_results_['params'][clf.best_index_]
        best_score = clf.best_score_
        return clf, best_parameters, best_score
    
    def classify(self, parameters, model, criteria):
        # model = Pipeline([('sampling', SMOTE()), ('classification', model)])
        # X_tr, X_te, y_tr, y_te = train_test_split(self.X, self.y, test_size=0.2, random_state=30, stratify=self.y)
        # clf, best_parameters, best_score = self.training_process(X_tr, y_tr, parameters, model, criteria)
        clf, best_parameters, best_score = self.training_process(self.X, self.y, parameters, model, criteria)
        print(best_parameters, best_score)
        # y_pred = clf.predict(X_te)
        # y_pred = clf.predict(self.X)
        # bacc, acc = balanced_accuracy_score(y_te, y_pred), accuracy_score(y_te, y_pred)
        # bacc, acc = balanced_accuracy_score(self.y, y_pred), accuracy_score(self.y, y_pred)
        return clf, best_parameters, best_score # , bacc, acc


# In[11]:


def f(colname, criteria):
    task = Classify()
    X, y = task.process_data(colname, data)
    print(X.columns, X.shape)
    task.summary()
    model_rf = Pipeline([
            ('random', RandomOverSampler(random_state = 1)),
            # ('sampling', SMOTE()),
            ('classification', RandomForestClassifier(random_state = 1))
        ])
    parameters_rf = {'classification__max_depth':[2,5,10], 
                  'classification__n_estimators':[50, 100, 200, 500, 1000], 
                  'classification__min_samples_split':[2]}
    
    model_log = Pipeline([
            ('random', RandomOverSampler(random_state = 1)),
            # ('sampling', SMOTE()),
            ('classification', LogisticRegression(multi_class = 'ovr', random_state = 1, fit_intercept = False))
        ])
    parameters_log = [
                    {'classification__penalty':['l2'], 
                      'classification__solver':['lbfgs'],
                      'classification__C':list(range(1, 11))},
                    {'classification__penalty':['l1'], 
                      'classification__solver':['liblinear'],
                      'classification__C':list(range(1, 11))}
                     ]
    model_gdbt = Pipeline([
        ('random', RandomOverSampler(random_state = 1)),
        # ('sampling', SMOTE()),
        ('classification', boosting(random_state = 1))
    ])
    parameters_gdbt = {'classification__max_depth':[2,5], 
                      'classification__n_estimators':[100, 500, 1000], 
                      'classification__min_samples_split': [2,4,6], 
                      'classification__learning_rate': [0.01, 0.1]}
    # X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=30, stratify=y)
    
    clf_log, best_parameters, best_score_log = task.classify(parameters_log, model_log, criteria) #training_process(X, y, parameters_log, model_log)
    clf_rf, best_parameters, best_score_rf = task.classify(parameters_rf, model_rf, criteria) # training_process(X, y, parameters_rf, model_rf)
    # clf_gdbt, best_parameters, best_score_gdbt = task.classify(parameters_gdbt, model_gdbt, criteria) # training_process(X, y, parameters_gdbt, model_gdbt)
    
    # prediction
    y_rf, y_log = clf_rf.predict(X), clf_log.predict(X)
    # y_gdbt = clf_gdbt.predict(X)
    res = {
        # 'gdbt': {'best_score': best_score_gdbt, 'balanced_acc': balanced_accuracy_score(y, y_gdbt), 'acc': accuracy_score(y, y_gdbt)},
        'rf': {'best_score': best_score_rf, 
               'balanced_acc': balanced_accuracy_score(y, y_rf), 
               'acc': accuracy_score(y, y_rf)},
        'log': {'best_score': best_score_log, 
               'balanced_acc': balanced_accuracy_score(y, y_log), 
               'acc': accuracy_score(y, y_log)}
          }
    '''
    clf_log, best_parameters, best_score_log, bacc_log, acc_log = task.classify(parameters_log, model_log, criteria) #training_process(X, y, parameters_log, model_log)
    clf_rf, best_parameters, best_score_rf, bacc_rf, acc_rf = task.classify(parameters_rf, model_rf, criteria) # training_process(X, y, parameters_rf, model_rf)
    clf_gdbt, best_parameters, best_score_gdbt, bacc_gdbt, acc_gdbt = task.classify(parameters_gdbt, model_gdbt, criteria) # training_process(X, y, parameters_gdbt, model_gdbt)
    res = {
        'gdbt': {'best_score': best_score_gdbt, 
               'balanced_acc': bacc_gdbt,
               'acc': acc_gdbt},
        'rf': {'best_score': best_score_rf, 
               'balanced_acc': bacc_rf, 
               'acc': acc_rf},
        'log': {'best_score': best_score_log, 
               'balanced_acc': bacc_log, 
               'acc': acc_log}
          }
    '''
    # print(best_score_gdbt, best_score_rf, best_score_log)
    return clf_log, clf_rf, res #clf_gdbt, res


# In[12]:


np.random.seed(1)
c = 'balanced_accuracy'
firstmile_train = f('firstmile_train', c)
firstmile_train[-1]
## random, smote, clf: 59, 54, 41
## smote, clf: 59, 55, 39
## random, clf: 58, 52, 42
## last_round: 45.76, 55.86, 61.28


# In[13]:


np.random.seed(1)
# c = 'f1_weighted'
c = 'balanced_accuracy'
lastmile_train = f('lastmile_train', c)
lastmile_train[-1]
# order: log - rf - gdbt (reverse)
## random, smote, clf: 41, 42, 38
## smote, clf: error
## random, clf: 38, 40, 39
# order: log - rf - gdbt
## last_round: 39.33, 39.04, 41.83


# In[14]:


np.random.seed(1)
c = 'balanced_accuracy'
lastmile_bus = f('lastmile_bus', c)
lastmile_bus[-1]
## random, smote, clf: 44, 50, 41
## smote, clf: 39, 48, 40
## random, clf: 43, 52, 40
## last_round: 40.75, 54.74, 45.23


# In[15]:


np.random.seed(1)
c = 'balanced_accuracy'
firstmile_bus = f('firstmile_bus', c)
firstmile_bus[-1]
## random, smote, clf: 37, 30, 28
## smote, clf: 31, 32, 29
## random, clf: 34, 34, 27
## last_round: 29.6, 32.8, 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




