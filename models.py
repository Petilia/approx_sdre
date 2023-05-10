import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

##########################
###### Monolit Models ####
##########################

class DummyModel2(nn.Module):
    def __init__(self, input_dim=6, 
                 output_dim=3, 
                 hidden_dim_1=16,
                 hidden_dim_2=16,
                 dropout_rate=0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, output_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x
    
class DummyModel3(nn.Module):
    def __init__(self, input_dim=6, 
                 output_dim=3, 
                 hidden_dim_1=16,
                 hidden_dim_2=16,
                 hidden_dim_3=16, 
                 dropout_rate=0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.fc4 = nn.Linear(hidden_dim_3, output_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = self.dropout(x)
        x = self.act(x)
        x = self.fc4(x)
        return x
    
##########################
#### ANFIS-like Models ###
##########################

### 1-Layer NN

class SplittedModelBlock(nn.Module):
    def __init__(self, input_dim=2,
                 output_dim=1, 
                 hidden_dim_1=16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, output_dim)
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class SplittedModel(nn.Module):
    def __init__(self, input_dim=2, 
                 output_dim=1, 
                 hidden_dim_1=16) -> None:
        super().__init__()
        self.block1 = SplittedModelBlock(input_dim=input_dim, 
                                         output_dim=output_dim, 
                                         hidden_dim_1=hidden_dim_1)
        self.block2 = SplittedModelBlock(input_dim=input_dim, 
                                         output_dim=output_dim, 
                                         hidden_dim_1=hidden_dim_1)
        self.block3 = SplittedModelBlock(input_dim=input_dim, 
                                         output_dim=output_dim, 
                                         hidden_dim_1=hidden_dim_1)

    def forward(self, x):
        x1 = x[:, (0, 3)]
        x2 = x[:, (1, 4)]
        x3 = x[:, (2, 5)]

        u1 = self.block1(x1)
        u2 = self.block1(x2)
        u3 = self.block1(x3)

        u = torch.cat((u1, u2, u3), dim=1)

        return u
    

### 2-Layer NN

class SplittedModelBlock2(nn.Module):
    def __init__(self, input_dim=2,
                 output_dim=1, 
                 hidden_dim_1=16, 
                 hidden_dim_2=16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, output_dim)
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x
    

class SplittedModel2(nn.Module):
    def __init__(self, input_dim=2, 
                 output_dim=1, 
                 hidden_dim_1=16,
                 hidden_dim_2=16) -> None:
        super().__init__()
        self.block1 = SplittedModelBlock2(input_dim=input_dim, 
                                         output_dim=output_dim, 
                                         hidden_dim_1=hidden_dim_1,
                                         hidden_dim_2=hidden_dim_2)
        self.block2 = SplittedModelBlock2(input_dim=input_dim, 
                                         output_dim=output_dim, 
                                         hidden_dim_1=hidden_dim_1,
                                         hidden_dim_2=hidden_dim_2)
        self.block3 = SplittedModelBlock2(input_dim=input_dim, 
                                         output_dim=output_dim, 
                                         hidden_dim_1=hidden_dim_1,
                                         hidden_dim_2=hidden_dim_2)

    def forward(self, x):
        x1 = x[:, (0, 3)]
        x2 = x[:, (1, 4)]
        x3 = x[:, (2, 5)]

        u1 = self.block1(x1)
        u2 = self.block1(x2)
        u3 = self.block1(x3)

        u = torch.cat((u1, u2, u3), dim=1)

        return u