

import numpy as np
#import scipy.special as ss
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
e = 3

class neuralNetwork :
    
    
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate) :
        
        self.inodes = inputnodes   #输入层单元数
        self.hnodes = hiddennodes  #隐藏层单元数
        self.onodes = outputnodes  #输出层单元数
        #self.kprob = keepprob      #准备dropout
     
                                   #层数=3，这个超参数已经确定，暂时不调
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.lr = learningrate     #两个参数矩阵的形状和数值初始化、学习率设定；
                                   #权重
        self.activation_function = lambda x : 1 / ( 1 + e**(-x) )
                                   #设定激活函数 
        pass
    
    
    def train(self, inputs_list, targets_list) :
        
        inputs = np.array(inputs_list, ndmin = 2).T    #输入神经元数量设定
        targets = np.array(targets_list, ndmin = 2).T  #输出神经元数量设定
        
        '''
        prob_mat1 = np.random.rand(self.wih.shape[0],self.wih.shape[1]) < self.kprob
        self.wih = self.wih * prob_mat1 + 0.001
        prob_mat2 = np.random.rand(self.who.shape[0],self.who.shape[1]) < self.kprob
        self.who = self.who * prob_mat2 + 0.001
        '''
        
        hidden_inputs = np.dot(self.wih, inputs)       #隐藏层输入数值计算
        hidden_outputs = self.activation_function(hidden_inputs) #隐藏层输出数值计算
        final_inputs = np.dot(self.who, hidden_outputs)#输出层输入数值计算
        final_outputs = self.activation_function(final_inputs)   #输出层输出数值计算
        
        output_errors = targets - final_outputs        #输出层误差计算
        hidden_errors = np.dot(self.who.T, output_errors)        #隐藏层误差计算
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),np.transpose(inputs))
                                                                 #参数更新一次
        pass
    
    def query(self, inputs_list) :
        
        inputs = np.array(inputs_list, ndmin = 2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs




input_nodes = 784
hidden_nodes = 200
output_nodes = 10
#keep_prob = 0.8
learning_rate = 0.2

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
training_data_file = open("C:/Users/sprin/Desktop/data_base/mnist_train.csv",'r')  #一个句柄，此后对文件操作都可以通过句柄完成
training_data_list = training_data_file.readlines()    #将文件里的所有行顺序读入training_data_list
training_data_file.close()                             #关闭文件

epochs = 5
for i in range(epochs):
    for record in training_data_list:     #每个training set里的数据都用于训练一次
    
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:])/255.0*0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass




test_data_file = open("C:/Users/sprin/Desktop/data_base/mnist_test.csv",'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

i2 = 0
#i3 = 0
right_number = 0
wrong_number = 0
#wrong_times = [0 for i in range(len(test_data_list)) ]

#while i3 < len(test_data_list):
#    wrong_times[i3] = None

while i2 < len(test_data_list):
    
    all_values = test_data_list[i2].split(',')
    print(all_values[0])

    image_array = np.asfarray(all_values[1:]).reshape((28,28))
    #plt.imshow(image_array, cmap = 'Greys', interpolation = 'None')

    m = n.query((np.asfarray(all_values[1:])/255.0 * 0.99) + 0.01)
    #print(m)
    guess = np.argmax(m,0)[0]
    print('This may be the number',guess)
    if int(guess) == int(all_values[0]):
        print('Right')
        right_number += 1
    else:
        print(i2,'Wrong')
        print(m)                       #只显示错误结果
       # i4 = 0
       # while i4 <= 100:
       #     if wrong_times[i4] != None:
       #         i4 += 1
       #         continue
       #     else:
       #         wrong_times[i4] = i2
       #         break
        wrong_number += 1
    i2 += 1
    
print('the precision is', right_number/(right_number + wrong_number)*100, 'percent')
#i5 = 0
#while i5 < len(test_data_list):
#    print(wrong_times[i5])

