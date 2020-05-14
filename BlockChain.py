# _*_ coding utf-8 _*_
"""
@File : BlockChain.py
@Author: yxwang
@Date : 2020/4/16
@Desc :
"""

import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import numpy as np

from stake_distribution import stake_distribution

'''
动作空间: 4维度，分别是验证节点选择，算法选择，区块大小， 时间间隔
状态空间: 单位时间间隔区块大小， 权益分布， 物理节点位置， 物理节点计算能力 
收益范围: 对不同的动作状态有不同的收益
'''

AREA_OF_NODES = (1000, 1000)
N_OF_NODES = 30
N_OF_BLOCK_PRODUCER = 21  # 常选择21
AVERAGE_TRANSACTION_SIZE = 200  # 200B
STAKE_OF_NODES = (10, 50)  # 对不同的节点权益分配,范围(0, 1)
COMPUTING_RESOURCE_OF_NODE = 0  # 暂定为10~30GHz
BLOCK_SIZE_LIMIT = 8 * 1024  # 最大区块的大小限制设定为8M，为统一单位定为B
MAX_BLOCK_INTERVAL = 10  # 最大区块间隔时间10s
ETA_S = 0.2
ETA_L = 0.3  # 基尼系数中对去中心化程度的最大限制设定为0.2与0.3
BATCH_SIZE = 3

'''
COMPUTING_RESOURCE_OF_NODE=0,BLOCK_SIZE_LIMIT = 8 * 1024  # 最大区块的大小限制设定为8M，为统一单位定为B
MAX_BLOCK_INTERVAL = 10  # 最大区块间隔时间10s
ETA_S = 0.2
ETA_L = 0.3  # 基尼系数中对去中心化程度的最大限制设定为0.2与0.3
BATCH_SIZE = 3
'''


class BlockChainEnv(gym.Env, utils.EzPickle):
    # Set this in SOME subclasses
    metadata = {'render.modes': []}
    reward_range = (-float('inf'), float('inf'))
    spec = None

    # Set these in ALL subclasses
    action_space = None
    observation_space = None

    def __init__(self, n_of_nodes=N_OF_NODES, area_of_nodes=AREA_OF_NODES,
                 average_transaction_size=AVERAGE_TRANSACTION_SIZE,
                 n_of_block_producer=N_OF_BLOCK_PRODUCER, stake_of_nodes=STAKE_OF_NODES,
                 computing_resource_of_node=COMPUTING_RESOURCE_OF_NODE,
                 block_size_limit=BLOCK_SIZE_LIMIT, max_block_interval=MAX_BLOCK_INTERVAL,
                 eta_s=ETA_S, eta_l=ETA_L, batch_size=BATCH_SIZE):
        utils.EzPickle.__init__(self, n_of_nodes, area_of_nodes, average_transaction_size, n_of_block_producer,
                                stake_of_nodes, computing_resource_of_node, block_size_limit, max_block_interval,
                                eta_s, eta_l, batch_size)
        # 定义环境
        self.n_of_nodes = n_of_nodes
        self.area_of_nodes = area_of_nodes
        self.average_transaction_size = average_transaction_size
        self.n_of_block_producer = n_of_block_producer
        self.stake_of_nodes = stake_of_nodes
        self.computing_resource_of_node = computing_resource_of_node
        self.block_size_limit = block_size_limit
        self.max_block_interval = max_block_interval
        self.eta_s = eta_s
        self.eta_l = eta_l
        self.batch_size = batch_size

        # self.observation_space = spaces.Dict({
        #     'average_block_size_per_interval': spaces.Box(low=0, high=200, shape=(1,)),
        #     'stake_of_nodes': spaces.Box(low=0, high=100, shape=(1,)),
        #     'geographical_of_IIoT_nodes': spaces.Box(low=np.array([-1000, -1000]), high=np.array([-1000, -1000])),
        #     'computing_capacity_of_IIoT_nodes': spaces.Box(low=10, high=30, shape=(1,))
        # })

        # 状态集合包括：吞吐量，中心化程度(权益和地理), 地理位置，计算能力，传输率，安全系数
        # OrderedDict([(k, space.sample()) for k, space in self.spaces.items()])
        self.observation_space = spaces.Dict(
            {
            'throughout': spaces.Box(low=0, high=np.inf, shape=(1,)),
            'G_gamma': spaces.Box(low=0, high=1, shape=(1,)),
            'G_lambda': spaces.Box(low=0, high=1, shape=(1,)),
            'geographical_of_nodes': spaces.Box(low=np.array([-1000, -1000]), high=np.array([-1000, -1000])),
            'computing_capacity_of_IIoT_nodes': spaces.Box(low=10, high=30, shape=(1,)),
            'transmission_rate': spaces.Box(low=10, high=100, shape=(1,)),
            'coef_of_security': spaces.Box(low=0, high=1, shape=(1,))
        })

        # self.action_space = spaces.Dict({
        #     'no_block_producer': spaces.Discrete(n_of_block_producer),
        #     'no_consensus_algorithm': spaces.Discrete(3),
        #     'block_size': spaces.Box(low=0, high=BLOCK_SIZE_LIMIT, shape=(1,)),
        #     'block_interval': spaces.Box(low=0, high=MAX_BLOCK_INTERVAL, shape=(1,))
        # })

        self.action_space = spaces.Dict({
            # 对所有的节点进行判断，值为1的节点是block_producer
            # 选择共识算法，将不同的共识算法记做0,1,2
            'no_block_producer': spaces.MultiBinary(n_of_nodes),
            'no_consensus_algorithm': spaces.Discrete(3),
            'block_size': spaces.Box(low=0, high=BLOCK_SIZE_LIMIT, shape=(1,)),
            'block_interval': spaces.Box(low=0, high=MAX_BLOCK_INTERVAL, shape=(1,))
        })

        self.seed()

        self.states = []
        self.actions = []
        self.state = None

    def reset(self):
        self.states = []
        self.actions = []
        pass

    def step(self, action):
        """
        在完成一幕之后，需要调用reset()重置环境状态
        接收到动作之后，返回一个元组，包括（观察到的状态， 收益， 是否结束， 信息）
        Arg:
            action: 通过智能体给出动作
        Returns:
            observation (object): 智能体观察到的当前状态
            reward (float) : 根据之前的动作所得到的的收益
            done (bool): 是否达到终止状态
            info (dict): 包括一些辅助信息
        """

        # 初始化状态
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        s = self.state

        alpha = 0.7
        beta = 0.5
        coef_of_security = self.get_security_coef()

        G_gamma, G_lambda = self.get_Gini()
        s_ = 0
        reward = 0
        done = False
        # average_block_size_per_interval, stake_of_nodes, geographical_of_IIoT_nodes, computing_capacity_of_IIoT_nodes \
        #     = (s[k] for k in s)

        G_gamma, G_lambda, coef_of_security, computing_capacity_of_IIoT_nodes, geographical_of_nodes, \
        throughout, transmission_rate = (s[k] for k in s)

        throughout = np.floor(action['block_size'] / self.average_transaction_size) / action['block_interval']
        stake_of_nodes = 0
        geographical_of_IIoT_nodes = action['no_block_producer']
        computing_capacity_of_IIoT_nodes = 0

        s_ = (
            self.average_transaction_size, stake_of_nodes, geographical_of_IIoT_nodes,
            computing_capacity_of_IIoT_nodes)

        self.state = (
            self.average_transaction_size, stake_of_nodes, geographical_of_IIoT_nodes,
            computing_capacity_of_IIoT_nodes)

        if not done and G_gamma < 1 and G_lambda < 1 and coef_of_security < 1:
            reward = beta * throughout + alpha * G_gamma + (1 - alpha) * G_lambda + (1 - beta) * coef_of_security
        else:
            reward = -1

        return s_, reward, done, {}

    def get_Gini(self):
        instance1 = stake_distribution(-self.area_of_nodes[0] / 2, self.area_of_nodes[0] / 2,
                                       -self.area_of_nodes[1] / 2, self.area_of_nodes[1] / 2, self.n_of_block_producer,
                                       stake=self.stake_of_nodes)
        return instance1.get_Gini_stake(), instance1.get_Gini_lambda()

    def get_security_coef(self):
        return 1

    # def render(self):
    #     """
    #     Renders the environment.
    #         The set of supported modes varies per environment. (And some
    #         environments do not support rendering at all.) By convention,
    #         if mode is:
    #         - human: render to the current display or terminal and
    #             return nothing. Usually for human consumption.
    #         - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
    #             representing RGB values for an x-by-y pixel image, suitable
    #             for turning into a video.
    #         - ansi: Return a string (str) or StringIO.StringIO containing a
    #             terminal-style text representation. The text can include newlines
    #             and ANSI escape sequences (e.g. for colors).
    #
    #         Note:
    #             Make sure that your class's metadata 'render.modes' key includes
    #             the list of supported modes. It's recommended to call super()
    #             in implementations to use the functionality of this method.
    #
    #         Args:
    #             mode (str): the mode to render with
    #
    #         Example:
    #
    #         class MyEnv(Env):
    #             metadata = {'render.modes': ['human', 'rgb_array']}
    #
    #             def render(self, mode='human'):
    #                 if mode == 'rgb_array':
    #                     return np.array(...) # return RGB frame suitable for video
    #                 elif mode == 'human':
    #                     ... # pop up a window and render
    #                 else:
    #                     super(MyEnv, self).render(mode=mode) # just raise an exception
    #     """
    #     pass

    def close(self):
        pass

    def seed(self):
        pass

    def sample(self):
        pass
