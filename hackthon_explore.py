# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:46:20 2020

@author: Sheeja Ayoob
"""
#import data
import pandas as pd
df=pd.read_csv(r"C:\Users\Sheeja Ayoob\Desktop\Train.csv")

#column names
mylist=list(df)

#no of rows and columns
rows=len(df.axes[0])
col=len(df.axes[1])

print("Number of Rows "+str(rows))
print("Number of Columns "+str(col))


#remove columns
#method 1
df1=df.iloc[:,6:]
#method 2
cols=[0,1,2,3,4,5]
df2=df.drop(df.columns[cols],axis=1)
df2=df.drop(df.columns[0:6],axis=1)

#no of abstracts under each subtopic
subtopics=list(df1)
counts=[]
for i in subtopics:
    counts.append((i,df1[i].sum()))
no_abs=pd.DataFrame(counts,columns=["topic","number"])
no_abs
#arranging in descending order
no_abs.sort_values("number",axis=0,ascending=False,inplace=True)



#no of abstracts with multilabels
#row sum
rowsum=df1.sum(axis=1)
rowsum_df=pd.DataFrame(rowsum,columns=["V"])
rowsum_df.V.unique()
#method1
row_list=rowsum_df["V"].tolist()
def countx(row_list, x):
    count = 0
    for ele in row_list:
        if (ele == x):
            count = count + 1
    return count

num=[1,2,3,4]
for i in num: 
    print('{} label has appeared in {} abstracts'.format(i, countx(row_list,i)))
#method 2       
labels = rowsum_df.pivot_table(index=['V'], aggfunc='size')
print (labels)

df4=pd.concat([df1,rowsum_df],axis=1)
df4.iat[14003,25]



sub=df4[df4.V==4]
index=sub.index
condition = sub["V"] == 4
indices =index[condition]
indices_list = indices.tolist()
print(indices_list)


k=df4.iloc[108:109,:]
c = k.columns[(k == 1).any()].tolist()
print (c)

k=[]
for i in indices_list:
    p=df4.iloc[i:(i+1),:]
    k.append(p)

    
   
    
c=k.columns[(k==1).any].tolist()
    
    