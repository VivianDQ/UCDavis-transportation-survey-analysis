#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import time
import collections

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)


# In[17]:


import statsmodels.discrete.discrete_model as sm


# In[18]:


class Classify:
    def __init__(self, col):
        # self.data = data
        self.data = None
        self.col = col
        
    def process_data(self, data):
        col = self.col
        xname = [# 'primary_role',
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
        
        data[col].replace(['bike'], 1, inplace = True)
        data[col].replace(['bus'], 2, inplace = True)
        data[col].replace(['drive'], 3, inplace = True)
        data[col].replace(['other'], 4, inplace = True)
        
        data = data.dropna(subset=[col] + xname + ['primary_role'])
        
        df1 = data[xname+[col]]
        self.data = pd.concat([df1, pd.get_dummies(data['primary_role'])], axis=1)
        # self.data = pd.get_dummies(data)
        return self.data
    
    def forward_selected(self):
        """Linear model designed by forward selection.

        Parameters:
        -----------
        data : pandas DataFrame with all possible predictors and response

        response: string, name of response column in data

        Returns:
        --------
        model: an "optimal" fitted statsmodels linear model
               with an intercept
               selected by forward selection
               evaluated by adjusted R-squared
        """
        response = self.col
        data = self.data
        remaining = set(data.columns)
        print(remaining)
        remaining.remove(response)
        selected = []
        current_score, best_new_score = float('Inf'), float('Inf')
        # print(remaining)
        while remaining and current_score == best_new_score:
            scores_with_candidates = []
            for candidate in remaining:
                l = sm.MNLogit(data[response].astype(int), data[selected+[candidate]].astype(int)).fit_regularized(penalty = 'l2')
                score = l.aic
                scores_with_candidates.append((score, candidate))
            scores_with_candidates.sort(reverse = True)
            best_new_score, best_candidate = scores_with_candidates.pop()
            # print(current_score, best_new_score)
            if current_score > best_new_score:
                remaining.remove(best_candidate)
                selected.append(best_candidate)
                current_score = best_new_score
        model = sm.MNLogit(data[response].astype(int), data[selected].astype(int)).fit_regularized(penalty = 'l2')
        return model


# In[19]:


# Load the data
# data_dic = pd.read_csv("data/cts1819_data_dictionary.csv")
data = pd.read_csv("data/CTS_noID_1819.csv")


# In[20]:


colname = 'firstmile_bus'
task = Classify(colname)
d = task.process_data(data)
print(d.shape)
firstmile_bus = task.forward_selected()


# In[21]:


colname = 'firstmile_train'
task = Classify(colname)
d = task.process_data(data)
print(d.shape)
firstmile_train = task.forward_selected()


# In[22]:


colname = 'lastmile_bus'
task = Classify(colname)
d = task.process_data(data)
print(d.shape)
lastmile_bus = task.forward_selected()


# In[23]:


colname = 'lastmile_train'
task = Classify(colname)
d = task.process_data(data)
print(d.shape)
lastmile_train = task.forward_selected()


# In[24]:


firstmile_bus.summary()


# In[25]:


firstmile_train.summary()


# In[26]:


lastmile_bus.summary()


# In[27]:


lastmile_train.summary()


# In[ ]:




