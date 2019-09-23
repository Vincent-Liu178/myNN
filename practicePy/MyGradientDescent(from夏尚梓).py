# my simple Gradient Descent
import numpy as np
import matplotlib.pyplot as plt


def h(x, a, b):
    return a * x + b  # 要拟合的函数

def cost(a, b, data_x, data_y):
    return 1.0/20 * np.sum((data_y - h(data_x, a, b))**2) # 误差计算



# 数据导入
data_x = np.arange(10).reshape((10,1))

 # 生成一个numpy.array并把它变（reshape）为10行1列的列向量；
 # numpy.array可以被当作向量，和一般的array不同。
 # np.arrange(10)就是生成1,2,……，10这样的一个数组。

noise = np.random.uniform(low=-2,high=2,size=(10,1))
 # 生成一些噪音；
 # np.random.uniform是从[low,high)采样，而且是均匀分布；
 # low和high都是float类型；size用来确定noise的维度：10行1列。

data_y = 2 * data_x + 1 + noise



# 参数初始化

a = -1
b = -20

# 画图
# matplotlib的显示模式默认为阻塞（block）模式，
# 因此若想动态显示图像，则需要使用交互（interactive）模式。
# 阻塞模式是指在程序中遇到plt.show()程序即停止，交互模式则会继续运行下去。

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(data_x, data_y)  # 蓝色散点是数据点 
lines = ax.plot(data_x,a * data_x + b ,c = 'r')
plt.ion()
plt.show()

for i in range(1000):
    a -= 0.015 * 0.1 * np.sum((h(data_x, a, b) - data_y) * data_x) # 梯度下降法参数a的更新
    b -= 0.015 * 0.1 * np.sum((h(data_x, a, b) - data_y) * 1)      # 梯度下降法参数b的更新
    if i % 50 == 0:
        print('cost=', cost(a, b, data_x, data_y))
        ax.lines.remove(lines[0])
        lines = ax.plot(data_x,a * data_x + b ,c = 'r')
        plt.pause(0.1)
print('a, b =',a,b)

