# _*_ coding utf-8 _*_
"""
@File : test.py
@Author: yxwang
@Date : 2020/4/30
@Desc :
"""

import numpy as np
from gym import spaces

from stake_distribution import stake_distribution

# stake = stake_distribution(-1, 1, -1, 1, 21, (10, 50))
# #
# # G_lambda = stake.get_Gini_lambda()
# #
# # print('G_lambda:{}'.format(G_lambda))
# #
# # G_gamma = stake.get_Gini_stake()
# #
# # print('G_gamma:{}'.format(G_gamma))

# action_space = spaces.Dict({
#     # 对所有的节点进行判断，值为1的节点是block_producer
#     # 选择共识算法，将不同的共识算法记做0,1,2
#     'no_block_producer': spaces.MultiBinary(4),
#     'no_consensus_algorithm': spaces.Discrete(3),
#     'block_size': spaces.Box(low=0, high=10, shape=(1,)),
#     'block_interval': spaces.Box(low=0, high=20, shape=(1,))})
#
# action = action_space.sample()
#
# a1, a2, a3, a4 = ((k, action[k]) for k in action)
# print('a1={}\na2={}\na3={}\na4={}'.format(a1, a2, a3, a4))
#
# observation_space = spaces.Dict({
#     'throughout': spaces.Box(low=0, high=np.inf, shape=(1,)),
#     'G_gamma': spaces.Box(low=0, high=1, shape=(1,)),
#     'G_lambda': spaces.Box(low=0, high=1, shape=(1,)),
#     'geographical_of_nodes': spaces.Box(low=np.array([-1000, -1000]), high=np.array([-1000, -1000])),
#     'computing_capacity_of_IIoT_nodes': spaces.Box(low=10, high=30, shape=(1,)),
#     'transmission_rate': spaces.Box(low=10, high=100, shape=(1,)),
#     'coef_of_security': spaces.Box(low=0, high=1, shape=(1,))})
#
# observation = observation_space.sample()
#
# print(observation)
#
# G_gamma, G_lambda, coef_of_security, computing_capacity_of_IIoT_nodes, geographical_of_nodes, \
# throughout, transmission_rate = (observation[k] for k in observation)
#
# print('G_gamma={}\nG_lambda={}\ncoef_of_security={}\ncomputing_capacity_of_IIoT_nodes={}\ngeographical_of_nodes={}\n'
#       'throughout={}\ntransmission_rate={}\n'.format(G_gamma, G_lambda, coef_of_security,
#                                                      computing_capacity_of_IIoT_nodes,
#                                                      geographical_of_nodes, throughout, transmission_rate))
# print(observation['geographical_of_nodes'])

# import exrex
# import regex as re
#
# env_id_re = re.compile(r'^(?:[\w:-]+\/)?([\w:.-]+)-v(\d+)$')
#
# id = 'wa/cartpole-v0'
#
# match = env_id_re.search(id)
#
# print(match.groups())

# import gym
#
# # create_env = lambda: gym.make('BlockChain-v0')
# # dummy_env = create_env()
# env = gym.make('BlockChain-v0')
# env.reset()
#
# for _ in range(3):
#     print(env.step(env.action_space.sample()))


# state = np.concatenate((np.random.uniform(low=0, high=1, size=(3,)), np.random.uniform(low=10, high=30, size=(1,)),
#                        np.random.uniform(low=10, high=50, size=(1,)), np.random.uniform(low=0, high=100, size=(1,))))
#
# print(np.array(state))

from geographical_coordination import geographical_coordination

instance1 = geographical_coordination(-1, 1, -1, 1)
gini_lambda = stake_distribution(2, (10, 20)).get_Gini_lambda(instance1)
print(gini_lambda)
