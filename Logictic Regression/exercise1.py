import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_classification.csv').values
N, d = data.shape
#data.sort()
x = data[:, 0:d-1].reshape(-1, d-1)
y = data[:, d-1].reshape(-1, 1)

true_x = []
true_y = []
false_x = []
false_y = []

for i in data:
    if i[d-1] == 1.:
        true_x.append(i[0])
        true_y.append(i[1])
    else:
        false_x.append(i[0])
        false_y.append(i[1])

plt.scatter(true_x, true_y, c='red', marker='o', edgecolors='none', s=30, label='Pass')
plt.scatter(false_x, false_y, c='blue', marker='x', edgecolors='none', s=30, label='not pass')
plt.legend(loc=1)
plt.xlabel('Số điểm')
plt.ylabel('Số giờ ngủ')
#plt.show()

x = np.hstack((np.ones((N, 1)), x))
w = np.array([0.,0.1,0.1]).reshape(-1, 1)

num = 1000
cost = np.zeros((num, 1))
learning_rate = 0.01

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

for i in range(1, num):
    y_predict = sigmoid(np.dot(x, w))
    cost[i] = -np.sum(np.multiply(y, np.log(y_predict)) + np.multiply(1-y, np.log(1-y_predict)))
    w -= learning_rate*np.dot(x.T, y_predict - y)

def predict(x1,x2):
    y1 = w[0] + x1*w[1] + x2*w[2]
    if(y1> -np.log(1/t-1)):
        print('pass')
    else:
        print('not pass')


t = 0.1
a=1
b=10
plt.plot((a,b), (-(w[0] + a*w[1] + np.log(1/t-1)) / w[2], -(w[0] + b*w[1] + np.log(1/t-1)) / w[2]), 'g')
#plt.savefig('exercise1.png')
plt.show()

predict(9,1)