#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import pandas as pd
df  = pd.read_csv("decision_Tree/train.csv")

data = pd.DataFrame(df, columns=['satisfaction_level', 'number_project', 'left'])
x=data['satisfaction_level']
y=data['number_project']
plot=data['left']
plt.xlabel('Satisfaction Level')
plt.ylabel('Project Number')
col= {0:'red', 1:'green'}
for i in range(1000):
    plt.scatter(x[i],y[i], color=col[plot[i]],s=3,label=col)

plt.show();

