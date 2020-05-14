# _*_ coding utf-8 _*_
"""
@File : stake_distribution.py
@Author: yxwang
@Date : 2020/4/30
@Desc :
"""

import numpy as np
from geographical_coordination import geographical_coordination


class stake_distribution:
    # def __init__(self, xMin, xMax, yMin, yMax, n_of_producer_nodes, stake, intensity=None):
    #     self.xMin = xMin
    #     self.xMax = xMax
    #     self.yMin = yMin
    #     self.yMax = yMax
    #     self.n_of_producer_nodes = n_of_producer_nodes
    #     self.intensity = intensity
    #     self.stake = stake
    #     self.g_gamma = []
    #     self.g_lambda = []
    #     self.G_gamma = None
    #     self.G_lambda = None

    def __init__(self, n_of_producer_nodes, stake, intensity=None):
        self.n_of_producer_nodes = n_of_producer_nodes
        self.intensity = intensity
        self.stake = stake
        self.g_gamma = []
        self.g_lambda = []
        self.G_gamma = None
        self.G_lambda = None

    def get_Gini_stake(self, instance_of_geo_coor):
        # geo_coor = geographical_coordination(self.xMin, self.xMax, self.yMin, self.yMax)
        xx = instance_of_geo_coor.geographical_coordinates()[0]
        self.n_of_producer_nodes = len(xx)

        self.g_gamma = np.random.normal(loc=np.floor(0.5 * (self.stake[0] + self.stake[1])), scale=10, size=xx.shape)

        self.g_gamma = self.g_gamma / self.g_gamma.sum()

        g0 = 0
        g1 = 0
        for i in range(self.n_of_producer_nodes):
            g0 += self.g_gamma[i]
            for j in range(self.n_of_producer_nodes):
                g = np.abs(self.g_gamma[i] - self.g_gamma[j])
                g1 += g
        self.G_gamma = g1 / (2 * self.n_of_producer_nodes * g0)
        return self.G_gamma

    def get_Gini_lambda(self, instance_of_geo_coor):
        # geo_coor = geographical_coordination(self.xMin, self.xMax, self.yMin, self.yMax)
        xx, yy, s = instance_of_geo_coor.geographical_coordinates()
        self.n_of_producer_nodes = len(xx)
        self.g_lambda = self.intensity_of_lambda(xx=xx, yy=yy, s=s)

        l0 = 0
        l1 = 0
        for i in range(self.n_of_producer_nodes):
            l1 += self.g_lambda[i]
            for j in range(self.n_of_producer_nodes):
                l = np.abs(self.g_lambda[i] - self.g_lambda[j])
                l0 += l
        self.G_lambda = l0 / (2 * self.n_of_producer_nodes * l1)
        return self.G_lambda

    def intensity_of_lambda(self, xx, yy, s):
        return 100 * np.exp(-(xx ** 2 + yy ** 2) / s ** 2)
