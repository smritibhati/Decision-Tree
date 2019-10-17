#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
eps = np.finfo(float).eps
from numpy import log2
import math
data=pd.read_csv('decision_Tree/train.csv')
data,testdata= np.split(data,[int(0.80*len(data))])


# In[25]:


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
                self.children,self.minsplit= partitionnumerical(rows,split)
            else:
                self.children= partitioncategorical(rows,split) #all children of this attribute in a dictionary
        print (self.value,"Node Value")
        return
        


# In[26]:


def maxfreq(rows):
    yes=0
    no=0
    for x in range(len(rows)):
        if rows.iloc[x]['left']==1:
            yes+=1
        else:
            no+=1
    return yes,no


# In[27]:


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
            ans+= float(val)/float(len(data1)) *(1-float(val)/float(len(data1)))
        
        ent+=ans*(len(data1)/len(data))
       
        ans=0
        for op in data2stats:
            val=data2stats[op]
            ans+= float(val)/float(len(data2)) *(1-float(val)/float(len(data2)))
        
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


# In[ ]:


# Partition function for numerical data while using misclassification


# In[28]:


def mispartitionnumerical(data, col):
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
            ans+= min(float(val)/float(len(data1)),(1-float(val)/float(len(data1))))
        
        ent+=ans*(len(data1)/len(data))
       
        ans=0
        for op in data2stats:
            val=data2stats[op]
            ans+= min(float(val)/float(len(data2)),(1-float(val)/float(len(data2))))
        
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


# In[29]:


#partitions the rows into multiple groups based on the column passed
def partitioncategorical(data,col):
    countdict={}
    uniquevalues = data[col].unique()
    for value in uniquevalues:
        countdict[value]= data[data[col]==value]
#     print (countdict)
    return countdict


# In[30]:


#matches the passed rows to count the number of yes and no in the rows 
def matchlabel(data):
    stats={}
    values, valuecount = np.unique(data['left'],return_counts=True)
    for i in range(len(values)):
        stats[values[i]]=valuecount[i]
    return stats


# In[31]:


def isnumeric(col):
    return data[col].dtype.kind in 'bifc'


# In[32]:


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
            ans+= float(value)/float(len(rowgroup)) * (1- float(value)/float(len(rowgroup)))
        
        entro+=ans*(len(rowgroup)/len(rows))
    return entro


# In[33]:


def misentropy(rows,col):
    entro=0
    
    if isnumeric(col):
#         print(col,"is numeric")
        countdict,minsplit= mispartitionnumerical(rows,col)
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
            ans+= min(float(value)/float(len(rowgroup)),(1- float(value)/float(len(rowgroup))))
        
        entro+=ans*(len(rowgroup)/len(rows))
    return entro


# In[34]:


def totalgini(data):
    countdict= partitioncategorical(data,'left')
    total = 0
    for key in countdict:
        value=len(countdict[key])
        total+= float(value)/float(len(data)) * (1-float(value)/float(len(data)))
#     print (total)
    return total


# In[35]:


totalgini(data)


# In[36]:


def totalmis(data):
    countdict=partitioncategorical(data,'left')
    total = 0
    for key in countdict:
        value=len(countdict[key])
        total+= min(float(value)/float(len(data)),(1-float(value)/float(len(data))))
#     print (total)
    return total


# In[37]:


totalmis(data)


# In[38]:


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
#         print ("infogain",infogain)
        if infogain>maxinfogain:
#             print(maxinfogain)
            maxinfogain=infogain
            attr=col
    
    print ("maxinfogain",maxinfogain)
    return maxinfogain,attr


# In[39]:


level=0
def buildTree(data):
#     global level
    total=totalgini(data)
#     print(total,"total")
    gain, split = informationgain(data,total)
    print("split", split)
#     level+=1
    if gain<=0:
        return Node(split,data,True,False)
    
    root = Node(split,data,False,isnumeric(split))
#     print( "root->children",root.children)
    for child in root.children:
#         print(root.children[child].drop(columns=[split]))
        if(isnumeric(split)):
            root.children[child]=buildTree(root.children[child])
        else:
            root.children[child]=buildTree(root.children[child].drop(columns=[split]))
        
#     level-=1
    return root
        


# In[40]:


root=buildTree(data)
# print(root.isleaf)


# In[41]:


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


# In[42]:


def calculate(fp,fn,tp,tn,wrong,correct):
    accuracy=correct/(wrong+correct)
    recall=tp/(tp+fn)
    precision=tp/(tp+fp)
    f1score=(2/(1/precision)+(1/recall))
    return accuracy,recall,precision,f1score


# In[43]:


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
    calculate(fp,fn,tp,tn,wrong,correct)


# In[44]:


predict(testdata)


# In[ ]:




