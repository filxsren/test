import torch
from torch.utils import data
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)


class CustomDataset_selfDefine(data.Dataset):
    def __init__(self,path):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lable =  torch.tensor(np.array(get_Dataset(path=path),dtype=np.float32)[0,1],dtype=torch.float32,device=device)
        self.data = torch.tensor(np.array(get_Dataset(path=path),dtype=np.float32)[1:],dtype=torch.float32,device=device)
        print(self.data.shape)
        # print(self.data.shape,self.lable.shape)
    def __getitem__(self, index):
        return self.data[index], self.lable[index]
    
    def __len__(self):
        return len(self.lable)
    
    def get_lable(self):
        return self.lable

def get_Dataset(path,mode='train'):
    # train_all=[]
    data_temp=pd.read_csv(path)
    # print(data_temp.info())
    
    data_temp['Age'].fillna(data_temp.Age.mean(),inplace=True)
    data_temp['Fare'].fillna(data_temp.Fare.mean(),inplace=True)
    data_temp=data_temp.drop('PassengerId',axis=1)
    data_temp=data_temp.drop('Name',axis=1)
    data_temp=data_temp.drop('Ticket',axis=1)
    data_temp=data_temp.drop('Cabin',axis=1)
    n=len(data_temp)
    for i in tqdm(range(n), desc="process_data"):
    # print(data_temp.info())
    # for i in range(len(data_temp)):
        # print(i)
        get_temp=data_temp.iloc[[i]]
        # print(get_temp)
        if get_temp.isnull().values.any():
            continue
        # print(get_temp)
        if get_temp.Sex.item()=='male':
            get_temp.loc[:,'Sex']=0
        elif get_temp.Sex.item()=='female':
            get_temp.loc[:,'Sex']=1
        # print(get_temp)
        if get_temp.Embarked.item()=='S':
            get_temp.loc[:,'Embarked']=0
        elif get_temp.Embarked.item()=='C':
            get_temp.loc[:,'Embarked']=1
        elif get_temp.Embarked.item()=='Q':
            get_temp.loc[:,'Embarked']=2
        
        data_temp.iloc[[i]]=get_temp
    data_temp.to_csv('train_temp.csv',index=False)
    print("finish process data")

        
        


    return data_temp.values.tolist()
get_Dataset(path='train.csv')



    



