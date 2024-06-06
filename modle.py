import torch
from torch.utils import data
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataSet import CustomDataset_selfDefine

#turn on the warning when debugging
import warnings

warnings.filterwarnings("ignore")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



test_data = CustomDataset_selfDefine("train.csv")
test_loader = DataLoader(dataset=test_data,batch_size=16,shuffle=True,drop_last=False)
train_data = CustomDataset_selfDefine("train.csv")
train_loader = DataLoader(dataset=test_data,batch_size=16,shuffle=True,drop_last=False)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(13, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x1 = F.relu(self.fc2(x))
        x = F.relu(self.fc3(torch.cat((x,x1),dim=1)))
        x = self.fc4(x)
        return x
net = Net().to(device)    
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

def train(mode):
    if mode =="train":
        loader=train_loader
        try:
            net.load_state_dict(torch.load("save.pt"))
            net.eval()
            print("load_state")
            n=40
        except:
            print("no save.pt")
            n=140
    elif mode=="test":
        loader=test_loader
        n=2


    
    losses = []    
    # for epoch in range(20):
    for i in tqdm(range(n), desc="training"):
        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # losses.append(loss.item())

            running_loss += loss.item()

            # print('[%d, %5d] loss: %.3f' % (epoch + 1, i+1, running_loss / 29))
            # running_loss = 0.0
        losses.append(running_loss / 29)
    torch.save(net.state_dict(), 'save.pt')
    print('Finished Training')
    plt.plot(losses, label='Loss')
    plt.title('Training Loss over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

train("train")