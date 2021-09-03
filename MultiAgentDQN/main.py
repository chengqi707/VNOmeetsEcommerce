import numpy as np
from tqdm import tqdm
import joblib
import random
import os

import torch
from torch.utils import tensorboard
import scipy.io as scio

from MultiAgent import MultiAgent
from Environment import Environment
from Replay import Replay


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


epsilon = 1.0
final_epsilon = 0.1
hysteretic = 0.2
final_hysteretic = 0.8
anneal_period = 15000
episodes = 20000
episode_length = 100
# channel_fixed_period = [10, 5, 2, 1]
reset_period = 10
train_interval = 50
num_agent = 1
target_update_frequency = 4
trace_length = 20
batch_size = 32
learning_rate=0.0001
gamma=0.95
memory_size = 1000
benchmark = 'random'

setup_seed(2333)

record_path = os.path.join('record', '{}_agents'.format(num_agent+1), benchmark, 'train')

env = Environment(num_EMs=10, num_MUs=1E+7, Q_0=30, Q_1=50, theta_m=50, h=1, w=0.2, A=1E+10, F_0=30,
                  numVNO=num_agent)
# env.channel_fixed_period = channel_fixed_period[0]

memory = Replay(num_VNO=num_agent, episode_length=episode_length, obs_size=env.observation_space,
                capacity=memory_size)

arena = MultiAgent(env, memory, num_agent=env.numVNO, episode_length=episode_length, trace_length=trace_length,
              batch_size=batch_size, learning_rate=learning_rate, epsilon=epsilon, final_epsilon=final_epsilon,
              gamma=gamma, hysteretic=hysteretic, final_hysteretic=final_hysteretic, anneal_period=anneal_period,
              training=True, dueling=True, benchmark=benchmark)

writer = tensorboard.SummaryWriter('./log')

loss_NO_list = []
reward_NO_list = []
loss_VNO_list = []
reward_VNO_list = []
for episode in tqdm(range(episodes)):
    if (episode + 1) % reset_period == 0:
        torch.cuda.empty_cache()
        arena.reset()
    loss = [0., 0.]
    for step in range(episode_length):
        #print(step)
        with torch.no_grad():
            arena.step()
        if (step + 1) % train_interval == 0:
            tempLoss = arena.train()
            loss[0] = tempLoss[0]
            loss[1] = tempLoss[1]

    arena.update_eps_hys()
    if (episode + 1) % target_update_frequency == 0:
        arena.update_target_model()

    loss_NO_list.append(loss[0])
    reward_NO_list.append(arena.episode_reward_NO)
    loss_VNO_list.append(loss[1])
    reward_VNO_list.append(arena.episode_reward_VNO)

    writer.add_scalar('NO\'s loss', loss[0], episode)
    writer.add_scalar('NO\'s reward', arena.episode_reward_NO, episode)
    writer.add_scalar('VNO\'s loss', loss[1], episode)
    writer.add_scalar('VNO\'s reward', arena.episode_reward_VNO, episode)

# save models
arena.save_models()
# save data
scio.savemat(os.path.join(record_path, 'NO\'s loss_proposed_{}.mat'.format(benchmark)), \
             {'loss_proposed_{}'.format(benchmark): loss_NO_list})
scio.savemat(os.path.join(record_path, 'NO\'s reward_proposed_{}.mat'.format(benchmark)), \
             {'reward_proposed_{}'.format(benchmark): reward_NO_list})
scio.savemat(os.path.join(record_path, 'VNO\'s loss_proposed_{}.mat'.format(benchmark)), \
             {'loss_proposed_{}'.format(benchmark): loss_VNO_list})
scio.savemat(os.path.join(record_path, 'VNO\'s reward_proposed_{}.mat'.format(benchmark)), \
             {'reward_proposed_{}'.format(benchmark): reward_VNO_list})

