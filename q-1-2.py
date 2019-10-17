#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
eps = np.finfo(float).eps
from numpy import log2
import math
# Divide the data into training and validation. Drop all non categorical columns from the dataframe
data=pd.read_csv('decision_Tree/train.csv')
data,testdata= np.split(data,[int(0.80*len(data))])


# In[21]:


# Node class is defines a node in the decision tree. Every node has the following properties:
#     isleaf : if the node is a leaf
#     isnumeric : if the node determines a numerical column
#     value : the name of the column on which this node is split
#     children : a dictionary of all the children nodes of this node
#     minsplit: if the node represents a numerical data, this is the value on which it is split


# In[22]:


class Node:
    def __init__(self,split,rows,leaf,numeric):
        self.isleaf=leaf
        self.rows=rows
        self.isnumeric=numeric
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
            if self.isnumeric:
                self.children,self.minsplit=partitionnumerical(rows,split)
            else:
                self.children= partitioncategorical(rows,split) #all children of this attribute in a dictionary
        print (self.value,"Node Value")
        return
        


# In[23]:


# Given a dataframe, counts the number of yes and nos labels
def maxfreq(rows):
    yes=0
    no=0
    for x in range(len(rows)):
        if rows.iloc[x]['left']==1:
            yes+=1
        else:
            no+=1
    return yes,no


# In[24]:


# Partitions numerical data in a binary fashion and returns a dictionary of its children snd the plit value.
# Works by finding he enropy for all the possible splits and then determinig the split which gives the minimum entropy
def partitionnumerical(data, col):
#     print("col",col)
    countdict={}
    minent=float('inf')
    data.sort_values([col],axis=0,ascending=True,inplace=True)
    uniquevalues = data[col].unique()
    for value in uniquevalues:
        ent=0
        data1=data[data[col]<=value]
        data2=data[data[col]>value]
        data1stats = matchlabel(data1)
        data2stats = matchlabel(data2)
        
        ans=0
        for op in data1stats:
            val=data1stats[op]
            ans-= float(val)/float(len(data1)) * math.log((float(val)/float(len(data1))),2)
        
        ent+=ans*(len(data1)/len(data))
       
        ans=0
        for op in data2stats:
            val=data2stats[op]
            ans-= float(val)/float(len(data2)) * math.log((float(val)/float(len(data2))),2)
        
        ent+=ans*(len(data2)/len(data))
    
        if ent<minent:
            minent=ent
            minsplit=value
    
#     print("categorical ent",minent)
#     print("minsplit",minsplit)
#     print("minsplit at",minsplit)
    countdict[0]=data[data[col]<=minsplit]
    countdict[1]=data[data[col]>minsplit]
    return countdict,minsplit


# In[25]:


#partitions the rows into multiple groups based on the column passed
def partitioncategorical(data,col):
    countdict={}
    uniquevalues = data[col].unique()
    for value in uniquevalues:
        countdict[value]= data[data[col]==value]
#     print (countdict)
    return countdict


# In[26]:


#matches the passed rows to count the number of yes and no in the rows 
def matchlabel(data):
    stats={}
    values, valuecount = np.unique(data['left'],return_counts=True)
    for i in range(len(values)):
        stats[values[i]]=valuecount[i]
    return stats


# In[44]:


#determines what columns are numeric 
def isnumeric(col):
    return data[col].dtype.kind in 'bifc'


# In[28]:


#calculate the entropy of the passed column
def entropy(rows,col):
    entro=0
    
    if isnumeric(col):
#         print(col,"is numeric")
        countdict,minsplit = partitionnumerical(rows,col)
    else:
        countdict= partitioncategorical(rows,col)
    
#     print("countdict",countdict)
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


# In[29]:


# calculates the total entropy of the data on the last column
def totalentropy(data):
    countdict=partitioncategorical(data,'left')
    total = 0
    for key in countdict:
        value=len(countdict[key])
        total-= float(value)/float(len(data)) * math.log((float(value)/float(len(data))),2)
    return total


# In[30]:


# calculates the total entropy of the data on the last column

def informationgain(data,total):
#     print("length of data", len(data))
    maxinfogain=0
    attr=0
    infogain=0
    for col in data:
        if col=='left': 
            continue
        ent=entropy(data,col)
#         print("Total and entropy",total,ent)
        infogain=total-ent
#         print (infogain)
        if infogain>maxinfogain:
#             print(maxinfogain)
            maxinfogain=infogain
            attr=col
    return maxinfogain,attr


# In[45]:


# A recursive funtion to build the tree.
# Initially called with the complete data.
# Recursive calls are made while the data is continuously partitiones and columns are dropped.
# Condition for leaf node: Gain<=0 or if there is only one label in the data

def buildTree(data):
    total=totalentropy(data)
    gain, split = informationgain(data,total)
    print("split", split)
    if gain<=0:
        return Node(split,data,True,False)
    
    root = Node(split,data,False,isnumeric(split))
    for child in root.children:
        if(isnumeric(split)):
            root.children[child]=buildTree(root.children[child])
        else:
            root.children[child]=buildTree(root.children[child].drop(columns=[split]))

    return root
        


# In[46]:


root=buildTree(data)
# print(root.isleaf)


# In[47]:


def findlabel(row):
    ptr = root
    while ptr.isleaf==False:
        if ptr.isnumeric==False:
            try:
                value=row[ptr.value]
#                 print(ptr.children)
                ptr=ptr.children[value]
            except:
                return 0
        else:
            try:
                value=row[ptr.value]
                if value<=ptr.minsplit:
                    ptr=ptr.children[0]
                else:
                    ptr=ptr.children[1]
            except:
                return 0
    
    return ptr.value


# In[48]:


def calculate(fp,fn,tp,tn,wrong,correct):
    accuracy=correct/(wrong+correct)
    recall=tp/(tp+fn)
    precision=tp/(tp+fp)
    f1score=(2/(1/precision)+(1/recall))
    return accuracy,recall,precision,f1score


# In[49]:


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
    print(fp,fn,tp,tn,wrong,correct)
    return calculate(fp,fn,tp,tn,wrong,correct)


# In[50]:


accuracy, recall, precision, F1score = predict(testdata)
print(accuracy,recall,precision,F1score)


# In[ ]:




