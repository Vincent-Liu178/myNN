
import numpy as np
import matplotlib
e = np.e

#一些样本
((x11,x12),y1) = ((0,0),1)
((x21,x22),y2) = ((0,1),0)
((x31,x32),y3) = ((1,0),0)
((x41,x42),y4) = ((1,1),1)

#向量化
x = [[x11,x12],
     [x21,x22],
     [x31,x32],
     [x41,x42]]
x = np.array(x)
y = [[y1],
     [y2],
     [y3],
     [y4]]
y = np.array(y)#array在神经网络中称张量

#激活函数
def g(z):
    return 1/(1+e**(-z))

#参数随机初始化
th1 = np.random.randn(2,2)
b1 = 0
#print(th1)
th2 = np.random.randn(2,1)
b2 = 0
#print(th2)
#th3 = np.random.randn(3,4)
#b3 = 0
#print(th3)
#th4 = np.random.randn(2,4)
#b4 = 0
#print(th4)

z1 = np.dot(x,th1) + b1
z1 = np.array(z1)
a1 = g(z1)
print(a1)

z2 = np.dot(a1,th2) + b2
z2 = np.array(z2)
a2 = g(z2)
print(a2)

de_3 = a2 - y
de_2 = (a2 * (1-a2)).T * np.dot(th2,de_3.T)  #th2是2*1，de_3是4*1
print(de_2)
#print(a2 * (1-a2))






