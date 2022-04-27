import numpy as np
import matplotlib.pyplot as plt

lam = 1.5  # 参数为1.5的泊松过程
n = 100  # 事件次数为10
# 10 个参数为 lam 的指数分布随机数
r = np.random.exponential(1 / lam, size=n)
# 10个事件的发生时刻
t = np.hstack([[0], np.cumsum(r)])
print(t)
for i in range(n):
    plt.plot((t[i], t[i+1]), (i, i), c='r')
plt.xlim([0, np.ceil(t.max())])
plt.ylim([0, n])
plt.show()