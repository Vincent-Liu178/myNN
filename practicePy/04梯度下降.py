import numpy as np
import matplotlib.pyplot as plt


def h( x, th0, th1 ):
    return th0 + th1 * x 

def cost( th0, th1, x, y ):
    m = x.shape[0]
    return 1/(2*m) * np.sum( (y - h( x, th0, th1 ))**2 )

def DcostDth0(th0, th1, x, y):
    m = x.shape[0]
    return -1/m * np.sum( ( y  - h( x, th0, th1 ) ) * 1 )

def DcostDth1(th0, th1, x, y):
    m = x.shape[0]
    return -1/m * np.sum( ( y  - h( x, th0, th1 ) ) * x )


#数据
x_data = np.arange(10).reshape(10,1)

noise = np.random.uniform( low = -2, high = 2, size = (10,1) )

y_data = -2 * x_data + 3 + noise

#参数随机初始化
the_random_th = np.random.rand(1,2)

th0 = the_random_th[0][0]
th1 = the_random_th[0][1]


#画图
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)  # 蓝色散点是数据点 
lines = ax.plot(x_data, th1 * x_data + th0 ,c = 'r')
plt.ion()
plt.show()



alpha = 0.01
for i in range(1000):
    th0 -= alpha * DcostDth0(th0, th1, x_data, y_data)
    th1 -= alpha * DcostDth1(th0, th1, x_data, y_data)
    if i %50 == 0:
         print('cost=', cost(th0, th1, x_data, y_data))
         ax.lines.remove(lines[0])
         lines = ax.plot(x_data,th1 * x_data + th0 ,c = 'r')
         plt.pause(0.1)
print('th1, th0 =',th1, th0)
