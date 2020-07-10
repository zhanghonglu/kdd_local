import numpy as np
import pandas as pd
from state import init_state
import data_process.action
from collections import deque
import matplotlib.pyplot as plt
import argparse
import os.path
import pickle
import torch
from ddqn_agent import Agent as Double_DQN_Agent
from dqn_agent import Agent as DQN_agent
from matplotlib.pyplot import MultipleLocator



def run(n_episodes=2000,  eps_start=1.0, eps_end=0.01, eps_decay=0.995,distance_limit = 2.5, time_limit=1000,order_num=10):
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):#周期开始
        print("第{}周期 初始化当前状态...".format(i_episode))
        state = init_state()#初始化最初状态
        score = 0

        while(state[0]<86400):#时间点没有超出范围
            print("周期{}进行学习 当前score{}".format(i_episode,score))
            print("当前状态：{}".format(state))
            print("获取当前状态可用动作")
            #获取当前可用订单
            state_actions = data_process.action.get_action(state[0],state[1],state[2],time_limit, distance_limit,order_num)
            # print(state_actions)
            if state_actions.values[0][1] == 0:
                break
            else:
                state_actions = state_actions.values
            #选择动作
            action_index = agent.act(state_actions, eps)
            action = state_actions[action_index]
            print(action)
            next_state = [action[4],action[7],action[8]]
            reward = action[9]
            action =action[3:]
            agent.step(state, action, reward, next_state)
            state = next_state
            score += reward
        print("当前周期结束")
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 1 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            my_path = os.path.abspath(os.path.dirname(__file__))
            file_name = os.path.join(my_path, 'model_dict/checkpoint_{}.pth'.format(i_episode))
            torch.save(agent.qnetwork_local.state_dict(), file_name)
            x= range(len(scores))
            plt.plot(x,scores)
            plt.ylabel('Score')
            plt.xlabel('Episode #')
            plt.xticks(x)
            for a, b in zip(x, scores):
                plt.text(a, b + 0.001, '%.4f' % b, ha='center', va='bottom', fontsize=9)
            plt.show()

    return scores


def parse_args():
    parser = argparse.ArgumentParser(description="run DQN models in gym env.")
    parser.add_argument('-model', type=str, default='Double_DQN',
                        help='DQN or Double_DQN or Dual_DQN or Dual_DDQN')
    parser.add_argument('-n_episodes', type=int, default=20000,
                        help='Number of episodes.')
    parser.add_argument('-max_t', type=int, default=1000,
                        help='max step in one episode.')
    parser.add_argument('-eps_start', type=float, default=1.0,
                        help='initial eps.')
    parser.add_argument('-eps_end', type=float, default=0.01,
                        help='min eps.')
    parser.add_argument('-eps_decay', type=float, default=0.9,
                        help='min eps.')
    # --------- learning args ----------------------#
    parser.add_argument('-buffer_size', type=int, default=int(1e5),
                        help='Number of samples in buff.')
    parser.add_argument('-batch_size', type=int, default=64,
                        help='batch size.')
    parser.add_argument('-gamma', type=float, default=0.90,
                        help='constant number for Q learning.')
    parser.add_argument('-tau', type=float, default=1e-3,
                        help='number of local network params moving to target network.')
    parser.add_argument('-lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('-update_steps', type=int, default=4,
                        help='step frequency for network updating')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if args.model == 'DQN':
        Agent = DQN_agent
        dual_network = False
    elif args.model == 'Double_DQN':
        Agent = Double_DQN_Agent
        dual_network = False
    elif args.model == 'Dual_DQN':
        Agent = DQN_agent
        dual_network = True
    else:
        Agent = Double_DQN_Agent
        dual_network = True

    agent = Agent(state_size=3, action_size=7, seed=0, lr=args.lr, buffer_size=args.buffer_size, batch_size=32,
                      update_step=args.update_steps, gamma=args.gamma, tau=args.tau, dual_network=dual_network)

    scores = run(n_episodes=2000,  eps_start=0.9, eps_end=0.05, eps_decay=0.85,distance_limit = 2.5, time_limit=1000,order_num=10)

    # plot the scores



