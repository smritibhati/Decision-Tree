#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import math
data=pd.read_csv('decision_Tree/train.csv')


# In[3]:


# Divide the data into training and validation. Drop all non categorical columns from the dataframe


# In[4]:


data,testdata= np.split(data,[int(0.80*len(data))])
data= pd.DataFrame(data,columns=['Work_accident', 'promotion_last_5years', 'sales', 'salary', 'left'])


# In[5]:


# Node class is defines a node in the decision tree. Every node has the following properties:
#     isleaf : if the node is a leaf
#     isnumeric : if the node determines a numerical column
#     value : the name of the column on which this node is split
#     children : a dictionary of all the children nodes of this node


# In[6]:


class Node:
    def __init__(self,split,rows,leaf):
        self.isleaf=leaf
        self.rows=rows
#         print("rows", rows)
        yes,no = maxfreq(rows)
       
        if yes==0 or no ==0:
            self.isleaf=True
        
        if self.isleaf==True:
            if yes>no:
                self.value=1
            else:
                self.value=0
            self.children={}
        else:
            self.value=split #name of the attribute is the value of the node
            self.children= partition(rows,split) #all children of this attribute in a dictionar
        print (self.value,"Node Value")
        return
        


# In[7]:


# Given a dataframe, counts the number of yes and nos labels


# In[8]:


def maxfreq(rows):
    yes=0
    no=0
    for x in range(len(rows)):
        if rows.iloc[x]['left']==1:
            yes+=1
        else:
            no+=1
    return yes,no


# In[9]:


# partitions the rows into multiple groups based on the column passed


# In[10]:


def partition(data,col):
    countdict={}
    uniquevalues = data[col].unique()
    for value in uniquevalues:
        countdict[value]= data[data[col]==value]
#     print (countdict)
    return countdict


# In[11]:


#matches the passed rows to count the number of yes and no in the rows 


# In[12]:


def matchlabel(data):
    stats={}
    values, valuecount = np.unique(data['left'],return_counts=True)
    for i in range(len(values)):
        stats[values[i]]=valuecount[i]
    return stats


# In[13]:


#calculate the entropy of the passed column


# In[14]:


def entropy(rows,col):
    entro=0
    countdict= partition(rows,col)
    for key in countdict:
        rowgroup=countdict[key]
        rowgroupstats=matchlabel(rowgroup)
        ans=0
#         print(rowgroupstats)
        for op in rowgroupstats:
            value=rowgroupstats[op]
            ans-= float(value)/float(len(rowgroup)) * math.log((float(value)/float(len(rowgroup))),2)
        
        entro+=ans*(len(rowgroup)/len(rows))
    return entro


# In[15]:


# calculates the total entropy of the data on the last column


# In[16]:


def totalentropy(data):
    countdict=partition(data,'left')
    total = 0
    for key in countdict:
        value=len(countdict[key])
        total-= float(value)/float(len(data)) * math.log((float(value)/float(len(data))),2)
    return total


# In[17]:


# Runs on all the columns of the data, and returns the max infogain and the column name for that max infogain


# In[18]:


def informationgain(data):
    global totalentropy
    print("length of data", len(data))
    maxinfogain=0
    attr=0
    infogain=0
    for col in data:
        if col=='left': 
            continue
        ent=entropy(data,col)
        
        infogain=totalentropy(data)-ent
        print (infogain)
        if infogain>maxinfogain:
#             print(maxinfogain)
            maxinfogain=infogain
            attr=col
    return maxinfogain,attr


# In[19]:


# A recursive funtion to build the tree.
# Initially called with the complete data.
# Recursive calls are made while the data is continuously partitiones and columns are dropped.
# Condition for leaf node: Gain<=0 or if there is only one label in the data


# In[20]:


def buildTree(data):
    global level
    gain, split = informationgain(data)
    print("split", split)
    if gain<=0:
        return Node(split,data,True)
    
    root = Node(split,data,False)
    for child in root.children:
        root.children[child]=buildTree(root.children[child].drop(columns=[split]))
        
    return root
        


# In[21]:


root=buildTree(data)
# print(root.isleaf)


# In[22]:


def findlabel(row):
    ptr = root
    while ptr.isleaf==False:
        value=row[ptr.value]
#         print(value)
        ptr=ptr.children[value]
    return ptr.value


# In[23]:


def calculate(fp,fn,tp,tn,wrong,correct):
    accuracy=correct/(wrong+correct)
    recall=tp/(tp+fn)
    precision=tp/(tp+fp)
    f1score=(2/(1/precision)+(1/recall))
    return accuracy,recall,precision,f1score


# In[24]:


def predict(data):
    correct=0
    wrong=0
    fp=0
    tp=0
    fn=0
    tn=0
    for i in range(0,len(data)):
        row=data.iloc[i]
        predictlabel=findlabel(row)
        if predictlabel==row['left']:
            if predictlabel==1:
                tp+=1
            else:
                tn+=1
            correct+=1
        else:
            if predictlabel==0:
                fn+=1
            else:
                fp+=1
            wrong+=1
#     print(fp,fn,tp,tn,wrong,correct)
    return (calculate(fp,fn,tp,tn,wrong,correct))


# In[26]:


accuracy, recall, precision, F1score = predict(testdata)
print(accuracy,recall,precision,F1score)


# In[ ]:




