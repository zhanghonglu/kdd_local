import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork, Dual_QNetwork
from data_process.action import get_action
import torch
import os.path
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    def __init__(self, state_size, action_size, seed, lr, buffer_size, batch_size, update_step,
                 gamma, tau, dual_network=False):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        if dual_network == False:
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        else:
            self.qnetwork_local = Dual_QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = Dual_QNetwork(state_size, action_size, seed).to(device)
        # self.qnetwork_local.load_state_dict(
        #     torch.load(os.path.join(os.path.abspath(os.path.dirname(__file__)), "model_dict/checkpoint_9.pth")))
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        self.update_step = update_step
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

    def step(self, state, action, reward, next_state):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state)

        self.t_step = (self.t_step + 1) % self.update_step
        print("当前历史记录缓冲{}".format(len(self.memory)))

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state_actions, eps):
        state_actions = torch.from_numpy(state_actions).float().to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_actions)
        self.qnetwork_local.train()
        # print(action_values)
        # Epsilon-greedy action.py selection
        print("选取动作：eps{}".format(eps))
        # print(np.arange(state_actions.shape[0]))
        # print(np.argmax(action_values.cpu().data.numpy()))
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())

        else:
            return random.choice(np.arange(state_actions.shape[0]))
    def learn(self, experiences, gamma):

        states, actions, rewards, next_states = experiences
        states_action = torch.cat([states, actions],dim=1)
        current_values = self.qnetwork_local(states_action)

        #target_action_id取的action集合
        next_states_actions = torch.tensor([get_action(i[0], i[1], i[2],order_num=10).values for i in next_states.cpu().data.numpy()]).cuda().float()
        target_action_idx = torch.tensor([torch.max(self.qnetwork_local(next_states_actions[i]), 0)[1][0] for i in range(self.batch_size) ])
        next_states_actions = next_states_actions.cpu().data.numpy()
        input_value = torch.from_numpy(np.asarray([next_states_actions[i][target_action_idx[i]] for i in range(self.batch_size)])).cuda().float()
        next_states_value = self.qnetwork_target(input_value)
        target_values = rewards + (gamma * next_states_value)

        # Compute loss
        loss = F.mse_loss(current_values, target_values)
        print("当前loss{}".format(loss))
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state):
        print("添加学习经历")
        e = self.experience(state, action, reward, next_state)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        return (states, actions, rewards, next_states)

    def __len__(self):
        return len(self.memory)