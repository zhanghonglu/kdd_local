import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_uints=64, fc2_uints=32,fc3_uints=8):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.norm = torch.nn.BatchNorm1d(num_features=state_size+action_size)
        self.fc1 = nn.Linear(state_size+action_size, fc1_uints)
        self.fc2 = nn.Linear(fc1_uints, fc2_uints)
        self.fc3 = nn.Linear(fc2_uints, fc3_uints)
        self.fc4 = nn.Linear(fc3_uints,1)

    def forward(self, state_action):
        x = self.norm(state_action)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x= self.fc4(x)
        return x

class Dual_QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_uints=32, fc2_uints=16):
        super(Dual_QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.feature = nn.Linear(state_size, fc1_uints)
        self.adventage_fc = nn.Linear(fc1_uints, fc2_uints)
        self.value_fc = nn.Linear(fc1_uints, fc2_uints)

        self.adventage_out = nn.Linear(fc2_uints, action_size)
        self.value_out = nn.Linear(fc2_uints, 1)

    def forward(self, state):
        feature = F.relu(self.feature(state))

        adventage = F.relu(self.adventage_fc(feature))
        adventage = self.adventage_out(adventage)

        value = F.relu(self.value_fc(feature))
        value = self.value_out(value)
        return value + adventage + adventage.mean()