import numpy as np
from tqdm import tqdm
import joblib
import random
import os

import torch
from torch.utils import tensorboard
import scipy.io as scio

from MultiAgentAsyn import MultiAgentAsyn
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
episodes = 25000
anneal_period = episodes * 3.0 / 4
episode_length = 100
reset_period = 2 * episodes
train_interval = 50
num_agent = 1
target_update_frequency = 4
trace_length = 20
batch_size = 32
learning_rate = 0.0001
gamma=0.95
memory_size = 1000
benchmark = 'random'

setup_seed(6789)

record_path = os.path.join('record', '{}_agents'.format(num_agent+1), benchmark, 'train')

env = Environment(num_EMs=10, num_MUs=1E+7, Q_0=30, Q_1=50, theta_m=50, h=1, w=0.275, A=1E+10, F_0=30,
                  numVNO=num_agent)
# env.channel_fixed_period = channel_fixed_period[0]

memory = Replay(num_VNO=num_agent, episode_length=episode_length, obs_size=env.observation_space,
                capacity=memory_size)

arena = MultiAgentAsyn(env, memory, num_agent=env.numVNO, episode_length=episode_length, trace_length=trace_length,
              batch_size=batch_size, learning_rate=learning_rate, epsilon=epsilon, final_epsilon=final_epsilon,
              gamma=gamma, hysteretic=hysteretic, final_hysteretic=final_hysteretic, anneal_period=anneal_period,
              training=True, dueling=False, benchmark=benchmark)

writer = tensorboard.SummaryWriter('./log')

loss_NO_list = []
reward_NO_list = []
loss_VNO_list = []
reward_VNO_list = []
data_plan_fee_of_VNO_list = []
price_of_ad_NO_list = []
price_of_data_NO_list = []
number_of_MU_NO_list = []
number_of_MU_VNO_list = []
number_of_ads_of_all_MUs_list = []
amount_of_data_purchased_by_VNO_list = []
number_of_ads_purchased_by_all_EMs_list = []
reward_in_each_episode_NO_list = []
reward_in_each_episode_VNO_list = []

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
    data_plan_fee_of_VNO_list.append(env.F_1)
    price_of_ad_NO_list.append(env.p)
    price_of_data_NO_list.append(env.c)
    number_of_MU_NO_list.append(env.N_0)
    number_of_MU_VNO_list.append(env.N_1)
    number_of_ads_of_all_MUs_list.append(env.G)
    amount_of_data_purchased_by_VNO_list.append(env.B_1)
    number_of_ads_purchased_by_all_EMs_list.append(env.m_total)
    reward_in_each_episode_NO_list.append(env.reward_NO)
    reward_in_each_episode_VNO_list.append(env.reward_VNO)

    writer.add_scalar('NO\'s loss', loss[0], episode)
    writer.add_scalar('NO\'s reward', arena.episode_reward_NO, episode)
    writer.add_scalar('VNO\'s loss', loss[1], episode)
    writer.add_scalar('VNO\'s reward', arena.episode_reward_VNO, episode)
    writer.add_scalar('data plan fee of VNO', env.F_1, episode)
    writer.add_scalar('ad price of VNO', env.p, episode)
    writer.add_scalar('NO\'s price of data', env.c, episode)
    writer.add_scalar('number_of_MU_NO', env.N_0, episode)
    writer.add_scalar('number_of_MU_VNO', env.N_1, episode)
    writer.add_scalar('number_of_ads_of_all_MUs', env.G, episode)
    writer.add_scalar('amount_of_data_purchased_by_VNO', env.B_1, episode)
    writer.add_scalar('number_of_ads_purchased_by_all_EMs', env.m_total, episode)
    writer.add_scalar('reward_in_each_episode_NO', env.reward_NO, episode)
    writer.add_scalar('reward_in_each_episode_VNO', env.reward_VNO, episode)

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



# """
# Evaluation
benchmark = 'random'
episode_len = 10
episode = 10
num_agent = 1
num_seeds = 20
Q_1_list = [40, 45, 50, 55, 60]
reward_NO_list = [[] for _ in range(len(Q_1_list))]
reward_VNO_list = [[] for _ in range(len(Q_1_list))]
reward_NO_benchmark_list = [[] for _ in range(len(Q_1_list))]
reward_VNO_benchmark_list = [[] for _ in range(len(Q_1_list))]

record_path = os.path.join('record', '{}_agents'.format(num_agent+1), benchmark, 'evaluate')

for i in tqdm(range(len(Q_1_list))):
    Q_1 = Q_1_list[i]

    reward_NO_mean = 0
    reward_VNO_mean = 0
    reward_NO_benchmark_mean = 0
    reward_VNO_benchmark_mean = 0

    for j in tqdm(range(num_seeds)):
        setup_seed(2345 + 5 * j)
        env = Environment(Q_1=Q_1, w=0.275, numVNO=num_agent)
        memory = None
        arena = MultiAgentAsyn(env, memory, num_agent=env.numVNO, episode_length=episode_length, trace_length=trace_length,
              batch_size=batch_size, learning_rate=learning_rate, epsilon=epsilon, final_epsilon=final_epsilon,
              gamma=gamma, hysteretic=hysteretic, final_hysteretic=final_hysteretic, anneal_period=anneal_period,
              training=False, dueling=False, benchmark=benchmark)
        reward_NO = 0
        reward_VNO = 0
        reward_NO_benchmark = 0
        reward_VNO_benchmark = 0
        for k in range(episode):
            for _ in range(episode_len):
                arena.step()
                reward_NO += env.reward_NO
                reward_VNO += env.reward_VNO
                reward_NO_benchmark += env.reward_NO_benchmark
                reward_VNO_benchmark += env.reward_VNO_benchmark

        reward_NO /= (episode_len * episode)
        reward_VNO /= (episode_len * episode)
        reward_NO_benchmark /= (episode_len * episode)
        reward_VNO_benchmark /= (episode_len * episode)

        reward_NO_mean += reward_NO / num_seeds
        reward_VNO_mean += reward_VNO / num_seeds
        reward_NO_benchmark_mean += reward_NO_benchmark / num_seeds
        reward_VNO_benchmark_mean += reward_VNO_benchmark / num_seeds

        reward_NO_list[i].append(reward_NO)
        reward_VNO_list[i].append(reward_VNO)
        reward_NO_benchmark_list[i].append(reward_NO_benchmark)
        reward_VNO_benchmark_list[i].append(reward_VNO_benchmark)

    print('\n')
    print('Q_1: {}'.format(Q_1))
    print('My method----Reward of NO: {:.2f}, Reward of VNO: {:.2f}'.format(reward_NO_mean, reward_VNO_mean))
    print('Benchmark method----Reward of NO: {:.2f}, Reward of VNO: {:.2f}'.format(reward_NO_benchmark_mean,
                                                                                  reward_VNO_benchmark_mean))
    print('\n')

# save data
scio.savemat(os.path.join(record_path, 'reward_of_NO_proposed_{}.mat'.format(benchmark)),
             {'reward_of_NO_proposed_{}'.format(benchmark): reward_NO_list})
scio.savemat(os.path.join(record_path, 'reward_of_VNO_proposed_{}.mat'.format(benchmark)),
             {'reward_of_VNO_proposed_{}'.format(benchmark): reward_VNO_list})
scio.savemat(os.path.join(record_path, 'reward_of_NO_benchmark_{}.mat'.format(benchmark)),
             {'reward_of_NO_benchmark_{}'.format(benchmark): reward_NO_benchmark_list})
scio.savemat(os.path.join(record_path, 'reward_of_VNO_benchmark_{}.mat'.format(benchmark)),
             {'reward_of_VNO_benchmark_{}'.format(benchmark): reward_VNO_benchmark_list})
# """
