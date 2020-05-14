# _*_ coding utf-8 _*_
"""
@File : geographical_coordination.py
@Author: yxwang
@Date : 2020/4/29
@Desc :
"""

from scipy.optimize import minimize
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt


# 1. 2D inhomogeneous PPP分布
# 2. 对于非其次泊松点过程的模拟，首先模拟一个均匀的泊松点过程，然后根据确定性函数适当地变换这些点
# 3. 模拟联合分布的随机变量的标准方法是使用  马尔可夫链蒙特卡洛；应用MCMC方法就是简单地将随机  点处理操作重复应用于所有点
#    将使用基于Thinning的通用但更简单的方法(Thinning是模拟非均匀泊松点过程的最简单，最通用的方法)

# plt.close('all')

class geographical_coordination:
    def __init__(self, xMin, xMax, yMin, yMax, num_Sim=1, s=0.5):
        self.xMin = xMin
        self.xMax = xMax
        self.yMin = yMin
        self.yMax = yMax
        self.xDelta = xMax - xMin
        self.yDelta = yMax - yMin
        self.areaTotal = self.xDelta * self.yDelta

        self.num_Sim = num_Sim
        self.s = s

        self.resultsOpt = None
        self.lambdaNegMin = None
        self.lambdaMax = None
        self.numbPointsRetained = None
        self.numbPoints = None

        self.xxRetained = []
        self.yyRetained = []
        self.xxThinned = []
        self.yyThinned = []

    # point process params
    def fun_lambda(self, x, y):
        # intensity function
        return 100 * np.exp(-(x ** 2 + y ** 2) / self.s ** 2)

    # define thinning probability function
    def fun_p(self, x, y):
        return self.fun_lambda(x, y) / self.lambdaMax

    def fun_neg(self, x):
        # negative of lambda
        # fun_neg = lambda x: -fun_lambda(x[0], x[1])
        return -self.fun_lambda(x[0], x[1])

    def geographical_coordinates(self):
        # initial value(ie center)
        xy0 = [(self.xMin + self.xMax) / 2, (self.yMin + self.yMax) / 2]

        # Find largest lambda value
        self.resultsOpt = minimize(self.fun_neg, xy0, bounds=((self.xMin, self.xMax), (self.yMin, self.yMax)))
        self.lambdaNegMin = self.resultsOpt.fun  # retrieve minimum value found by minimize
        self.lambdaMax = -self.lambdaNegMin

        # for collecting statistics -- set num_Sim=1 for one simulation
        self.numbPointsRetained = np.zeros(self.num_Sim)

        for ii in range(self.num_Sim):
            # Simulate a Poisson point process
            # Poisson number of points
            self.numbPoints = np.random.poisson(self.areaTotal * self.lambdaMax)
            # x coordinates of Poisson points
            # y coordinates of Poisson points
            xx = np.random.uniform(0, self.xDelta, (self.numbPoints, 1)) + self.xMin
            yy = np.random.uniform(0, self.yDelta, (self.numbPoints, 1)) + self.yMin

            # calculate spatially-dependent thinning probabilities
            p = self.fun_p(xx, yy)

            # Generate Bernoulli variables (ie coin flips) for thinning
            # points to be retained
            # Spatially independent thinning
            booleRetained = np.random.uniform(0, 1, (self.numbPoints, 1)) < p
            booleThinned = ~booleRetained

            # x/y locations of retained points
            self.xxRetained = xx[booleRetained]
            self.yyRetained = yy[booleRetained]
            self.xxThinned = xx[booleThinned]
            self.yyThinned = yy[booleThinned]
            self.numbPointsRetained[ii] = self.xxRetained.size

        return self.xxRetained, self.yyRetained, self.s


