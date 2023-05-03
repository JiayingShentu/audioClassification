import random

import numpy as np
from matplotlib import pyplot as plt


def gauss_noisy(x, y):
    """
    对输入数据加入高斯噪声
    :param x: x轴数据
    :param y: y轴数据
    :return:
    """
    mu = 0
    sigma = 0.05
    for i in range(len(x)):
        x[i] += random.gauss(mu, sigma)
        y[i] += random.gauss(mu, sigma)


if __name__ == '__main__':
    # 在0-5的区间上生成50个点作为测试数据
    xl = np.linspace(0, 5, 50, endpoint=True)
    yl = np.sin(xl)

    # 加入高斯噪声
    gauss_noisy(xl, yl)

    # 画出这些点
    plt.plot(xl, yl, linestyle='', marker='.')
    plt.show()