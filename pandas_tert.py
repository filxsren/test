import torch
from torch.utils import data
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd

path='train.csv'
data_temp=pd.read_csv(path)
print(data_temp.info())

data_temp['Age'].fillna(data_temp.Age.mean(),inplace=True)
data_temp['Fare'].fillna(data_temp.Fare.mean(),inplace=True)
data_temp=data_temp.drop('PassengerId',axis=1)
data_temp=data_temp.drop('Name',axis=1)
data_temp=data_temp.drop('Ticket',axis=1)
data_temp=data_temp.drop('Cabin',axis=1)

print(data_temp.info())
# for i in range(len(data_temp)):
#     get_temp=data_temp.iloc[[i]]
#     print(get_temp)
#     if get_temp.isnull().values.any():
#         continue
#     print(get_temp)
#     data_temp.iloc[[i]]=get_temp
i=0
get_temp=data_temp.iloc[[i]]
print(get_temp)
# if get_temp.isnull().values.any():
#     continue
print(get_temp.Sex=='male')
if get_temp.Sex.item()=='male':
    get_temp.loc[0,'Sex']=0
elif get_temp.Sex.item()=='female':
    get_temp.loc[0,'Sex']=1
print(get_temp)
data_temp.iloc[[i]]=get_temp
data_temp.to_csv('train_temp.csv',index=False)
