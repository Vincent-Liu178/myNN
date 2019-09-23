#一个画图的例子


import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3,3,50)
y1 = 2*x + 1
m = x**2

plt.figure()

plt.plot(x,y1,color = 'red',linewidth = 1.0,linestyle = '--')
plt.plot(x,m)

plt.xlim((-1,2))
plt.ylim((-2,3))
plt.xlabel('i am x')
plt.ylabel('i am y')

#new_ticks = np.linspace(-1,2,5)
#print(new_ticks)
#plt.xticks(np.linspace(-1,2,5))
plt.yticks([-1,0,1],
          [r'$medium\ \alpha\ well$','well','well done'])


ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))


plt.show()


